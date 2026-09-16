"""召回训练：阶段一（确定性双塔）与阶段二（高斯双塔）共用同一套数据与循环。

数据构造（时间序，防泄漏是重点）：

    对每个用户，按时间把他的交互切成 训练 / 验证 / 测试；
    预测第 i 个正样本时，**学生需求文本只能用前 i 个交互构造**——
    否则「需求」里含有了待预测的那门课本身，等于把答案抄进输入。

两种样本模式：
    `all_prefixes`：每个交互都作正样本（信号最多，5 万样本时约 700s/epoch）
    `last_prefix` ：每用户只用最后一个交互作正样本（快约 10 倍，更贴近线上形态）

负样本来自 batch 内其它资源的向量（in-batch），零构造开销；
可选再拼入若干热门资源，抑制「全推热门」的偏置。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch

from ...features.encoder import Encoder, resource_text
from .gaussian import GaussianTwoTower
from .semantic import TwoTowerSemantic, fuse_quality, normalize_minmax, rank_percentile

PREFIX_SEP = " "


@dataclass
class Interaction:
    """一次交互：user / item 均为数据集原始 ID，text 是该资源的文本。"""

    user: int
    item: int
    ts: int
    text: str = ""


@dataclass
class RecallExamples:
    """训练样本集合（`pos` 是 `item_ids` 里的下标）。"""

    need_texts: list[str] = field(default_factory=list)
    pos: list[int] = field(default_factory=list)
    user_of: list[int] = field(default_factory=list)
    history_len: list[int] = field(default_factory=list)   # 构造该需求用了多少条历史

    def __len__(self) -> int:
        return len(self.pos)


def build_interactions(bundle) -> list[Interaction]:
    """把 DataBundle 转成带文本的交互列表。

    文本只取自资源的标题+正文——**不含用户侧信息**，因此不同用户对同一资源得到同一文本，
    资源编码可以全局缓存复用。
    """
    text_of = {r.resource_id: resource_text(r) for r in bundle.resources}
    return [Interaction(user=b.user_id, item=b.resource_id, ts=b.ts,
                        text=text_of.get(b.resource_id, ""))
            for b in bundle.behaviors]


def build_examples(interactions: list[Interaction], item_index: dict[int, int],
                   max_need_items: int = 20, min_history: int = 1,
                   mode: str = "all_prefixes") -> RecallExamples:
    """按时间序构造「用历史需求预测下一门课」的样本。

    `mode="all_prefixes"`：每个用户第 i 个交互（0 基）都作正样本，需求文本由前 i 个交互拼成；
    `mode="last_prefix"`：每个用户只用**最后一个交互**作正样本，需求由其余全部历史拼成。

    两种模式都保证：正样本自身绝不出现在需求文本里。
    """
    by_user: dict[int, list[Interaction]] = {}
    for it in interactions:
        if it.item in item_index:
            by_user.setdefault(it.user, []).append(it)

    ex = RecallExamples()
    for user, acts in by_user.items():
        acts.sort(key=lambda x: (x.ts, x.item))
        if mode == "last_prefix":
            if len(acts) <= min_history:
                continue
            history = [it.text for it in acts[:-1]]
            ex.need_texts.append(PREFIX_SEP.join(history[-max_need_items:]))
            ex.pos.append(item_index[acts[-1].item])
            ex.user_of.append(user)
            ex.history_len.append(min(len(history), max_need_items))
            continue
        if mode != "all_prefixes":
            raise ValueError(f"未知的样本构造模式: {mode!r}")
        history: list[str] = []
        for pos_idx, act in enumerate(acts):
            if pos_idx >= min_history:
                ex.need_texts.append(PREFIX_SEP.join(history[-max_need_items:]))
                ex.pos.append(item_index[act.item])
                ex.user_of.append(user)
                ex.history_len.append(min(len(history), max_need_items))
            history.append(act.text)
    return ex


def popular_negatives(interactions: list[Interaction], item_index: dict[int, int],
                      top_n: int) -> list[int]:
    """被交互最多的资源下标：热门显式负样本。"""
    counts = Counter(it.item for it in interactions)
    return [item_index[i] for i, _ in counts.most_common() if i in item_index][:top_n]


def build_model(kind: str, in_dim: int, config):
    """按 `tower_kind` 构造塔：deterministic（阶段一）或 gaussian（阶段二）。"""
    if kind == "deterministic":
        return TwoTowerSemantic(
            in_dim=in_dim, out_dim=config.recall_embed_dim,
            hidden=config.recall_hidden_dim, dropout=config.recall_dropout,
            temperature=config.recall_temperature)
    if kind == "gaussian":
        return GaussianTwoTower(
            in_dim=in_dim, out_dim=config.recall_embed_dim,
            hidden=config.recall_hidden_dim, dropout=config.recall_dropout,
            temperature=config.recall_temperature,
            variance_floor=getattr(config, "gaussian_variance_floor", 0.05),
            max_std=getattr(config, "gaussian_max_std", 10.0),
            history_signal=getattr(config, "gaussian_history_signal", True))
    raise ValueError(f"未知的 tower_kind: {kind!r}（可选 deterministic / gaussian）")


def train_tower(encoder: Encoder, interactions: list[Interaction], item_ids: list[int],
                config, kind: str | None = None, device: str = "cpu",
                verbose: bool = False):
    """训练召回塔。`kind` 默认取 `config.tower_kind`。

    `item_ids` 是全部资源 ID（升序），其文本编码构成资源塔的输入空间；
    两个塔作用在同一编码器输出上，因此资源编码只需算一次。
    """
    kind = kind or getattr(config, "tower_kind", "deterministic")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    item_index = {item: i for i, item in enumerate(item_ids)}
    examples = build_examples(interactions, item_index,
                              max_need_items=config.recall_max_need_items,
                              min_history=1,
                              mode=getattr(config, "recall_sample_mode", "last_prefix"))
    if len(examples) == 0:
        raise ValueError("没有可训练的样本：每个用户至少需要 2 次交互")

    text_of = {}
    for it in interactions:
        text_of.setdefault(it.item, it.text)
    corpus = [text_of.get(item, "") for item in item_ids]
    encoder.fit(corpus)

    if verbose:
        print(f"[recall:{kind}] 样本 {len(examples)}，语料 {len(corpus)} 条，"
              f"编码维度 {encoder.dim}")

    with torch.no_grad():
        item_embs = torch.tensor(encoder.encode(corpus), dtype=torch.float32, device=device)
        need_embs = torch.tensor(encoder.encode(examples.need_texts),
                                 dtype=torch.float32, device=device)
    pos_idx = torch.tensor(examples.pos, dtype=torch.long, device=device)
    hist_len = torch.tensor(examples.history_len, dtype=torch.long, device=device)

    model = build_model(kind, int(item_embs.size(1)), config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.recall_lr)

    neg_embs = None
    if config.recall_neg_items > 0:
        hot = popular_negatives(interactions, item_index, config.recall_neg_items)
        if hot:
            neg_embs = item_embs[torch.tensor(hot, dtype=torch.long, device=device)]

    n = len(examples)
    rng = np.random.default_rng(config.seed)
    loss_history: list[float] = []
    for epoch in range(config.recall_epochs):
        perm = rng.permutation(n)
        total, steps = 0.0, 0
        for start in range(0, n, config.recall_batch_size):
            batch = perm[start:start + config.recall_batch_size]
            if len(batch) < 2:
                continue          # in-batch 负样本至少要有 2 条才有对比信号
            b_idx = torch.tensor(batch, dtype=torch.long, device=device)
            loss = model(need_embs[b_idx], item_embs[pos_idx[b_idx]],
                         neg_emb=neg_embs, tau=config.recall_temperature,
                         history_len=hist_len[b_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
        avg = total / max(1, steps)
        loss_history.append(avg)
        if verbose:
            print(f"[recall:{kind}] epoch {epoch + 1}/{config.recall_epochs} loss={avg:.4f}")

    model.loss_history = loss_history       # 供调用方断言「可学习」
    model.tower_kind = kind
    model.encoder_version = encoder.version
    return model.cpu()


def train_two_tower(encoder: Encoder, interactions: list[Interaction], item_ids: list[int],
                    config, device: str = "cpu", verbose: bool = False):
    """阶段一（确定性双塔）训练入口；保留旧名字以便既有调用与测试不变。"""
    return train_tower(encoder, interactions, item_ids, config,
                       kind="deterministic", device=device, verbose=verbose)


def train_gaussian_two_tower(encoder: Encoder, interactions: list[Interaction],
                             item_ids: list[int], config, device: str = "cpu",
                             verbose: bool = False):
    """阶段二（高斯双塔）训练入口。"""
    return train_tower(encoder, interactions, item_ids, config,
                       kind="gaussian", device=device, verbose=verbose)


# --- 推理辅助 ---

@torch.no_grad()
def prepare_item_matrix(model, encoder: Encoder, item_texts: list[str],
                        device: str = "cpu") -> None:
    """离线预计算资源表示并挂到模型上：推理时只剩学生塔一次前向。

    两种塔都缓存均值矩阵 `item_embs_cache`；高斯塔额外缓存标准差 `item_std_cache`。
    """
    embs = torch.tensor(encoder.encode(item_texts), dtype=torch.float32, device=device)
    if isinstance(model, GaussianTwoTower):
        mu, std = model.encode_items(embs)
        model.item_embs_cache = mu
        model.item_std_cache = std
    else:
        model.item_embs_cache = model.encode_items(embs).to(device)


@torch.no_grad()
def score_users(model, encoder: Encoder, need_texts: list[str],
                device: str = "cpu",
                history_len: list[int] | None = None) -> np.ndarray:
    """给一批学生需求打分全部资源，返回 (n_users, n_items) 分数矩阵。

    两种塔走同一套代码：确定性塔把自己的点表示当成 σ≈0 的退化分布，
    因此这里统一按「负的 W₂²」计算，确定性塔下等价于内积（差一个常数）。
    """
    if not need_texts:
        n_items = model.item_embs_cache.size(0)
        return np.zeros((0, n_items), dtype=np.float32)
    need_embs = torch.tensor(encoder.encode(need_texts), dtype=torch.float32, device=device)
    hist = (torch.tensor(history_len, dtype=torch.long, device=device)
            if history_len is not None else None)

    mu_u, std_u = model.student_dist(need_embs, hist)
    mu_i = model.item_embs_cache
    std_i = getattr(model, "item_std_cache", None)
    if std_i is None:
        # 确定性塔：只有均值，直接内积（与 -W₂² 的排序完全一致）
        return (mu_u @ mu_i.T).cpu().numpy()
    mean_term = ((mu_u.unsqueeze(1) - mu_i.unsqueeze(0)) ** 2).sum(-1)
    cov_term = ((std_u.unsqueeze(1) - std_i.unsqueeze(0)) ** 2).sum(-1)
    return (-(mean_term + cov_term)).cpu().numpy()


def fuse_scores(semantic: np.ndarray, ratings: np.ndarray, popularity: np.ndarray,
                config) -> np.ndarray:
    """把语义/距离分数与资源质量信号融合（CourseHub 的复合打分函数）。

    评分用 min-max（分布有上下界的 1~5 分），热度用秩百分位（零交互占多数的长尾计数）。
    """
    return fuse_quality(
        semantic,
        normalize_minmax(ratings),
        rank_percentile(popularity),
        w_semantic=config.quality_w_semantic,
        w_rating=config.quality_w_rating,
        w_popularity=config.quality_w_popularity,
    )
