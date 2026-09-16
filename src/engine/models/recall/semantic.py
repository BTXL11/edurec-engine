"""阶段一：双塔语义召回（CourseHub 式确定性嵌入）。

思路对应论文 CourseHub：给每个资源与学生需求各建一个嵌入，用余弦相似度衡量距离，
再融合质量信号（评分、热度）做最终排序。与 CourseHub 的差别在于：

  - CourseHub 直接把查询文本的编码当最终表示，**不做训练**（论文也把「未做个性化用户建模」
    列为主要局限）；这里把学生需求编码投影成**可学习的用户塔**，用交互数据训练，
    使「需求 → 资源」的对齐能力超过纯文本相似度；
  - 学生需求文本由**行为历史反推**（平台没有查询/学习目标字段），而不是实时查询。

链路：

    资源文本 ──Encoder──► e_i ──ItemTower(线性/MLP)──► item 向量 ──┐
                                                                   ├─ 内积打分
    学生行为历史 ──拼文本──►Encoder──► e_u ──StudentTower──► user 向量 ┘

两个塔都作用在**同一编码器空间**上，因此资源向量可离线预计算，学生侧推理只需一次前向。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TwoTowerSemantic(nn.Module):
    """语义双塔：把编码器空间映射到共享的召回空间。

    `normalize=True`（默认）时输出单位向量，内积即余弦相似度——与 CourseHub 的度量一致；
    训练时用 InfoNCE，温度 `tau` 控制分布锐度。
    """

    def __init__(self, in_dim: int, out_dim: int = 128, hidden: int = 256,
                 dropout: float = 0.0, normalize: bool = True, temperature: float = 0.05):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize = normalize
        self.temperature = temperature
        self.student_tower = _mlp([in_dim, hidden, out_dim], dropout)
        self.item_tower = _mlp([in_dim, hidden, out_dim], dropout)

    def _finalize(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x, dim=-1) if self.normalize else x

    def student_emb(self, need_emb: torch.Tensor) -> torch.Tensor:
        """学生需求编码 → 需求向量 (B, out_dim)。"""
        return self._finalize(self.student_tower(need_emb))

    def item_emb(self, resource_emb: torch.Tensor) -> torch.Tensor:
        """资源编码 → 资源向量 (B, out_dim)。"""
        return self._finalize(self.item_tower(resource_emb))

    def score(self, need_emb: torch.Tensor, resource_emb: torch.Tensor) -> torch.Tensor:
        """成对打分 (B,)：两个塔的内积。"""
        return (self.student_emb(need_emb) * self.item_emb(resource_emb)).sum(-1)

    def forward(self, need_emb: torch.Tensor, pos_emb: torch.Tensor,
                neg_emb: torch.Tensor | None = None,
                tau: float | None = None) -> torch.Tensor:
        """InfoNCE 损失：batch 内其它资源的向量作为负样本。

        logits[i, j] = <user_i, item_j> / tau，目标是对角线。这与 Word2Vec 的
        sampled softmax 同源：正样本对确定，负样本直接取 batch 内其它 item，
        零构造开销、实现简单。
        """
        tau = self.temperature if tau is None else tau
        ue = self.student_emb(need_emb)                       # (B, D)
        pe = self.item_emb(pos_emb)                           # (B, D)
        logits = ue @ pe.T / tau                              # (B, B)
        if neg_emb is not None and neg_emb.numel() > 0:
            ne = self.item_emb(neg_emb)                       # (N, D)
            logits = torch.cat([logits, ue @ ne.T / tau], dim=-1)
        target = torch.arange(logits.size(0), device=logits.device)
        return nn.functional.cross_entropy(logits, target)

    @torch.no_grad()
    def encode_items(self, resource_embs: torch.Tensor,
                     batch_size: int = 4096) -> torch.Tensor:
        """离线预计算全部资源向量（分块，避免大矩阵一次性占满内存）。"""
        self.eval()
        out = []
        for start in range(0, resource_embs.size(0), batch_size):
            chunk = resource_embs[start:start + batch_size]
            out.append(self.item_emb(chunk))
        return torch.cat(out, dim=0) if out else torch.zeros(0, self.out_dim)

    @torch.no_grad()
    def rank(self, need_emb: torch.Tensor, item_matrix: torch.Tensor,
             top_k: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """给学生需求排序全部资源，返回 (分数, 下标) 按分数降序。

        一期是**全量点积**，不建 ANN 索引：资源量小（几百~几千）时更简单也更精确，
        资源量大后把 `item_matrix` 交给 faiss 即可，接口不变。
        """
        self.eval()
        ue = self.student_emb(need_emb)                       # (B, D) or (D,)
        if ue.dim() == 1:
            ue = ue.unsqueeze(0)
        scores = ue @ item_matrix.T                           # (B, n_items)
        if top_k is not None:
            scores, idx = torch.topk(scores, k=min(top_k, scores.size(1)), dim=1)
        else:
            idx = torch.argsort(scores, dim=1, descending=True)
            scores = torch.gather(scores, 1, idx)
        return scores, idx


# --- 质量信号融合（CourseHub 式） ---

def normalize_minmax(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """min-max 归一到 [0, 1]；全相同或为空时返回全 0。

    适用于**分布大致对称且有上下界**的信号，例如课程评分（1~5 分）。
    不适合「大多数取值为 0」的长尾计数——那种情形用 `rank_percentile`。
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    out = values
    if not higher_is_better:
        out = -out
    lo, hi = float(out.min()), float(out.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (out - lo) / (hi - lo)


def normalize_log(values: np.ndarray) -> np.ndarray:
    """对数压缩后按最大值归一：把长尾热度压成可比区间（CourseHub 的 popularity 归一）。

    保留 `log1p(0) = 0` 的下界，因此「零交互」不会被当成中等热度。
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    logged = np.log1p(np.maximum(values, 0.0))
    hi = float(logged.max())
    return logged / hi if hi > 1e-12 else np.zeros_like(logged)


def rank_percentile(values: np.ndarray) -> np.ndarray:
    """秩百分位归一到 [0, 1]：并列取值共享平均名次，最大值为 1、最小值为 0。

    比 min-max 更适合零值占多数的长尾计数：`[0, 0, 0, 5, 100]` 归一为
    `[0.25, 0.25, 0.25, 0.75, 1.0]`（三个 0 并列，共享中间百分位），
    而不是把零值当成「最低档但非零」。
    另有一个实际优势：它只依赖名次，因此某种信号整体失效（例如全站热度都是 0）
    时会返回**全 0**，不会给每门课加一个常数去稀释其它信号。
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    if values.size == 1 or float(values.max() - values.min()) < 1e-12:
        return np.zeros(values.size, dtype=np.float32)     # 无区分度 → 不给分
    ranks = _average_ranks(values)                         # 1 基平均秩
    return ((ranks - 1.0) / (values.size - 1.0)).astype(np.float32)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0        # 名次 i+1 … j+1 的平均
        i = j + 1
    return ranks


def fuse_quality(semantic: np.ndarray, rating: np.ndarray, popularity: np.ndarray,
                 w_semantic: float = 0.7, w_rating: float = 0.2,
                 w_popularity: float = 0.1) -> np.ndarray:
    """score = w1·语义相似度 + w2·归一化评分 + w3·归一化热度。

    CourseHub 用网格搜索在验证集上定权重，最优为 (0.7, 0.2, 0.1)，这里沿用其默认值。
    缺信号的场景自动退化：MOOCCube 没有评分，把 `w_rating` 置 0 即得纯语义+热度。
    """
    total = w_semantic + w_rating + w_popularity
    if total <= 0:
        raise ValueError("质量融合权重之和必须为正")
    return (w_semantic * semantic + w_rating * rating
            + w_popularity * popularity) / total
