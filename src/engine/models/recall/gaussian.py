"""阶段二：概率式（高斯）双塔召回。

阶段一把学生需求与资源都表示成空间里的**一个点**，用内积/余弦衡量距离。
但一个点表达不了两件事：

  1. **需求本身的不确定性**——历史只有 1~2 门课的学生，他的「需求」其实很模糊；
  2. **兴趣的多峰性**——一个学生可能同时对两三个方向感兴趣，单点表示被迫取平均。

阶段二把表示升级为**分布**：`N(μ, Σ)`。均值是「主要需求/内容主题」，
方差是「这件事有多确定」。参考：

  - **UKT**（Uncertainty-aware Knowledge Tracing）：用均值 + 协方差嵌入表示学生状态，
    并用 **2-Wasserstein 距离**衡量两个分布之间的差异（点积无法度量分布差异）；
  - **U-GLAD**：用方差**校准期望** `ĥ = μ ⊙ (1 − σ²)`——不确定性高时压低均值，
    使表示更保守、更稳健。

实现取舍（都是为了让一期代码量可控且可验证）：

  - **对角协方差**：`Σ = diag(σ²)`。完整协方差需要矩阵平方根与特征分解，
    而在对角假设下 2-Wasserstein 距离有闭式解，且与「每个维度一个不确定度」的直觉一致；
  - **在单位球面上取分布**：均值 L2 归一化，于是均值项 `‖μ_u − μ_i‖²` 落在 [0, 4]，
    与阶段一的余弦度量同一量纲；
  - **得分用负距离**：`score = -W₂²`，越大越好，可直接套用阶段一的排序与评估代码。

**⚠️ 在 MOOCCube 上的实测结论：本扩展不产生收益，默认不启用。**

（`tower_kind="deterministic"` 是默认值；本模块保留为可选实现与可复现的负面证据。）

实测数据（5000 活跃用户 / 706 门课 / 1 vs 99 采样候选）：

  ===================  ==========  ===========  =========  =========
  方法                  HitRate@5   HitRate@10   NDCG@10    MRR
  ===================  ==========  ===========  =========  =========
  阶段一（确定性）       0.2864      0.4432       0.1710     0.1947
  阶段二（高斯）         0.2694      0.4256       0.1594     0.1843
  热门排序               0.3586      0.4930       0.2174     0.2509
  ===================  ==========  ===========  =========  =========

诊断出的**根因**是方差没有被学出区分度：

  - **资源侧**方差几乎恒定：均值 0.5019、标准差 0.0013，变异系数仅 **0.26%**。
    于是 W₂ 的协方差项在排序中近似常数，等于没有贡献；
  - **需求侧**方差与历史长度确实相关（相关系数 +0.40），但**方向相反**——
    历史越长方差反而越大，与「历史短=不确定」的假设不符；
  - 方差正则项只保证方差**不塌缩**，它无法制造「历史短 → 方差大」这种结构；
    训练目标里也没有任何一项要求方差去解释不确定性。

结论：概率式表示要真正生效，需要**方差有可学的信息来源**。UKT/U-GLAD 之所以可行，
是因为它们有交互**序列**（时序建模能暴露「行为矛盾/稀疏」），而本任务里学生需求只是
**一段拼接文本**，文本本身不含「历史有多短/多矛盾」的信息——仅把历史长度当标量特征
补进去（`history_signal=True`）也不足以让方差承担有意义的角色。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .semantic import _mlp


class GaussianTwoTower(nn.Module):
    """高斯双塔：两个塔各自输出均值与方差，用 2-Wasserstein 距离打分。

    `score(u, i) = -W₂²(N(μ_u, Σ_u), N(μ_i, Σ_i))`，其中

        W₂² = ‖μ_u − μ_i‖² + Σ_d (σ_u,d − σ_i,d)²        （对角协方差下的闭式解）

    第二项是「不确定性之差」：两边的方差越接近，距离越小。这意味着

      * 模型可以把「确定」表示成小方差、把「模糊」表示成大方差；
      * 一个高方差的冷启动需求，与「各方差都很大」的热门宽泛内容更近，
        从而自然地被推向保守/热门推荐——这正是 U-GLAD 想要的效果。
    """

    def __init__(self, in_dim: int, out_dim: int = 128, hidden: int = 256,
                 dropout: float = 0.0, temperature: float = 0.05,
                 min_std: float = 1e-3, max_std: float = 10.0,
                 variance_floor: float = 0.05, history_signal: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.temperature = temperature
        self.min_std = min_std
        self.max_std = max_std
        self.variance_floor = variance_floor
        self.history_signal = history_signal

        # 历史长度信号：让「需求有多模糊」这件事**进入输入**。
        # 实测教训：只把方差作为可学参数时，模型在 MOOCCube 上学不出任何不确定性
        # （方差恒为 0.505，与历史长短无关），校准因此纯粹起负作用。
        # UKT/U-GLAD 能从交互序列里推断不确定性，而这里的学生需求只是一段拼起来的文本，
        # 文本本身不含「历史有多短」的信息，必须显式补进去。
        extra = 1 if history_signal else 0
        # 每个塔输出 2*out_dim：前半是均值，后半是「标准差」的原始值
        self.student_tower = _mlp([in_dim + extra, hidden, 2 * out_dim], dropout)
        self.item_tower = _mlp([in_dim, hidden, 2 * out_dim], dropout)

    @staticmethod
    def history_feature(history_len: torch.Tensor) -> torch.Tensor:
        """把历史长度压成 [0,1] 附近的标量特征：`log1p(n) / log(21)`。

        用对数是因为「1 条 → 2 条」与「19 条 → 20 条」对确定性的影响并不等价。
        """
        return (torch.log1p(history_len.float()) / torch.log(torch.tensor(21.0))).unsqueeze(-1)

    def _student_input(self, need_emb: torch.Tensor,
                       history_len: torch.Tensor | None) -> torch.Tensor:
        if not self.history_signal:
            return need_emb
        squeezed = need_emb.dim() == 1
        if squeezed:
            need_emb = need_emb.unsqueeze(0)          # 允许传单个需求向量
        if history_len is None:
            history_len = torch.ones(need_emb.size(0), device=need_emb.device)
        feat = self.history_feature(history_len).to(need_emb.dtype)
        if feat.dim() == 1:
            feat = feat.unsqueeze(-1).expand(need_emb.size(0), 1)
        out = torch.cat([need_emb, feat], dim=-1)
        return out.squeeze(0) if squeezed else out

    # --- 分布参数 ---

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, raw = x.chunk(2, dim=-1)
        mu = nn.functional.normalize(mu, dim=-1)          # 单位球面上的均值
        # softplus 保证正值；再夹到 [min_std, max_std] 避免方差爆炸或塌缩
        std = nn.functional.softplus(raw).clamp(self.min_std, self.max_std)
        return mu, std

    def student_dist(self, need_emb: torch.Tensor,
                     history_len: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """学生需求分布 (μ_u, σ_u)。

        `history_len` 是构造该需求所用的历史交互数；开启 `history_signal` 时它是**必需**的
        输入（缺省按 1 条处理）。它表达「需求有多模糊」，是方差能从数据里学出来的前提。
        """
        return self._split(self.student_tower(self._student_input(need_emb, history_len)))

    def item_dist(self, resource_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """资源分布 (μ_i, σ_i)。"""
        return self._split(self.item_tower(resource_emb))

    # --- 距离与打分 ---

    @staticmethod
    def wasserstein2(mu_a: torch.Tensor, std_a: torch.Tensor,
                     mu_b: torch.Tensor, std_b: torch.Tensor) -> torch.Tensor:
        """对角协方差下的 2-Wasserstein **平方**距离，逐对计算。

        对完整协方差一般式 `tr(Σ_a + Σ_b − 2(Σ_a^{1/2}Σ_bΣ_a^{1/2})^{1/2})`，
        在 `Σ = diag(σ²)` 时中间的矩阵平方根退化为逐元素乘积，
        整体化简为 `Σ_d (σ_a,d − σ_b,d)²`。
        """
        mean_term = ((mu_a - mu_b) ** 2).sum(dim=-1)          # ‖μ_a − μ_b‖²
        cov_term = ((std_a - std_b) ** 2).sum(dim=-1)         # Σ_d (σ_a,d − σ_b,d)²
        return mean_term + cov_term

    def score(self, need_emb: torch.Tensor, resource_emb: torch.Tensor,
              history_len: torch.Tensor | None = None) -> torch.Tensor:
        """逐对打分：负的 W₂²（越大越相似）。"""
        mu_u, std_u = self.student_dist(need_emb, history_len)
        mu_i, std_i = self.item_dist(resource_emb)
        return -self.wasserstein2(mu_u, std_u, mu_i, std_i)

    # --- U-GLAD 式方差校准 ---

    def calibrated_mean(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """`ĥ = μ ⊙ (1 − mean(σ²))`：不确定性越高，均值被压得越低。

        U-GLAD 用逐维 `μ ⊙ (1 − σ²)`；这里改用**标量化的平均方差**做校准，
        因为逐维写法需要向量与索引对齐、在本任务里语义不明确（UKT 原文也是对
        `diag(·)` 操作）。标量版本的含义清楚：整体越不确定，需求表示越保守。
        """
        uncertainty = (std ** 2).mean(dim=-1, keepdim=True)     # (B, 1)
        return mu * (1.0 - uncertainty).clamp(min=0.0)

    def student_emb(self, need_emb: torch.Tensor,
                    history_len: torch.Tensor | None = None,
                    calibrated: bool = False) -> torch.Tensor:
        """学生需求的单向量表示：默认取分布的均值，`calibrated=True` 时取校准后的均值。

        提供这个方法是让高斯塔与阶段一（[`TwoTowerSemantic`]）共用同一套排序/评估接口。
        """
        mu, std = self.student_dist(need_emb, history_len)
        return self.calibrated_mean(mu, std) if calibrated else mu

    # --- 训练 ---

    def forward(self, need_emb: torch.Tensor, pos_emb: torch.Tensor,
                neg_emb: torch.Tensor | None = None,
                tau: float | None = None,
                history_len: torch.Tensor | None = None) -> torch.Tensor:
        """InfoNCE 损失：距离越近 logits 越大，batch 内其它资源为负样本。"""
        tau = self.temperature if tau is None else tau
        mu_u, std_u = self.student_dist(need_emb, history_len)  # (B, D)
        mu_p, std_p = self.item_dist(pos_emb)                   # (B, D)

        # (B, B)：负距离 / τ
        mean_term = ((mu_u.unsqueeze(1) - mu_p.unsqueeze(0)) ** 2).sum(-1)
        cov_term = ((std_u.unsqueeze(1) - std_p.unsqueeze(0)) ** 2).sum(-1)
        logits = -(mean_term + cov_term) / tau
        if neg_emb is not None and neg_emb.numel() > 0:
            mu_n, std_n = self.item_dist(neg_emb)              # (N, D)
            mean_n = ((mu_u.unsqueeze(1) - mu_n.unsqueeze(0)) ** 2).sum(-1)
            cov_n = ((std_u.unsqueeze(1) - std_n.unsqueeze(0)) ** 2).sum(-1)
            logits = torch.cat([logits, -(mean_n + cov_n) / tau], dim=-1)

        target = torch.arange(logits.size(0), device=logits.device)
        loss = nn.functional.cross_entropy(logits, target)
        if self.variance_floor > 0:
            loss = loss + self.variance_floor * self.variance_regularizer(std_u, std_p)
        return loss

    @staticmethod
    def variance_regularizer(std_u: torch.Tensor, std_p: torch.Tensor) -> torch.Tensor:
        """阻止方差塌缩到 `min_std`。

        若不加约束，模型会让所有方差趋同，W₂ 的第二项失去作用、退化成阶段一。
        这里惩罚「平均对数方差」`-log σ² / 2`（等价于 `log(1/σ)`）：

          * σ = 1 时该项为 0；
          * σ 趋近 0 时该项 → +∞，因此**最小化它会阻止塌缩**；
          * σ 放大时该项变负，因此也不鼓励无意义地放大方差。

        注意符号：`log(σ²)` 本身在 σ<1 时是负数，直接加进 loss 会**鼓励**塌缩，
        所以这里取的是它的相反数。
        """
        return -0.5 * ((std_u ** 2).log().mean() + (std_p ** 2).log().mean())

    # --- 不确定性诊断 ---

    @torch.no_grad()
    def uncertainty_report(self, need_emb: torch.Tensor,
                           history_len=None,
                           item_emb: torch.Tensor | None = None) -> dict[str, float]:
        """诊断方差是否真的携带了信息——这是概率式扩展能否成立的前提。

        `history_len` 可以是 tensor 或普通整数列表（内部会转换）。

        判据：
          * `item_var_cv`：资源方差的变异系数。接近 0 说明 W₂ 的协方差项在排序里
            近似常数，等于没起作用（MOOCCube 上实测约 **0.0026**）；
          * `history_corr`：需求方差与历史长度的相关系数（Pearson，只表示方向与强度，
            对非线性关系会低估强度）。理想的概率建模应当是**负相关**
            （历史越短越不确定）；若接近 0 或为正，说明方差没有学到「数据稀疏度」。
            实测 MOOCCube 上为 **+0.40（方向相反）**。
        """
        if history_len is not None and not torch.is_tensor(history_len):
            history_len = torch.tensor(list(history_len), dtype=torch.long)
        result: dict[str, float] = {}
        if item_emb is not None:
            _, std_i = self.item_dist(item_emb)
            var = (std_i ** 2).mean(dim=-1)
            mean = float(var.mean())
            result["item_var_mean"] = mean
            result["item_var_cv"] = float(var.std() / mean) if mean > 1e-12 else 0.0

        mu_u, std_u = self.student_dist(need_emb, history_len)
        var_u = (std_u ** 2).mean(dim=-1)
        result["student_var_mean"] = float(var_u.mean())
        if history_len is not None and history_len.numel() > 1:
            h = history_len.float()
            v = var_u.float()
            h_centered = h - h.mean()
            v_centered = v - v.mean()
            denom = float((h_centered.norm() * v_centered.norm()))
            result["history_corr"] = (float((h_centered * v_centered).sum()) / denom
                                      if denom > 1e-12 else 0.0)
        return result

    # --- 与阶段一接口对齐（复用同一套排序/评估代码） ---

    @torch.no_grad()
    def encode_items(self, resource_embs: torch.Tensor,
                     batch_size: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
        """离线预计算全部资源的 (μ, σ)。"""
        self.eval()
        mus, stds = [], []
        for start in range(0, resource_embs.size(0), batch_size):
            chunk = resource_embs[start:start + batch_size]
            mu, std = self.item_dist(chunk)
            mus.append(mu)
            stds.append(std)
        if not mus:
            empty = torch.zeros(0, self.out_dim)
            return empty, empty.clone()
        return torch.cat(mus, dim=0), torch.cat(stds, dim=0)

    @torch.no_grad()
    def rank(self, need_emb: torch.Tensor, item_matrix,
             top_k: int | None = None,
             history_len: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """给需求排序全部资源；`item_matrix` 这里是 `(μ_i, σ_i)` 二元组。"""
        self.eval()
        mu_i, std_i = item_matrix
        mu_u, std_u = self.student_dist(need_emb, history_len)
        if mu_u.dim() == 1:
            mu_u, std_u = mu_u.unsqueeze(0), std_u.unsqueeze(0)
        mean_term = ((mu_u.unsqueeze(1) - mu_i.unsqueeze(0)) ** 2).sum(-1)
        cov_term = ((std_u.unsqueeze(1) - std_i.unsqueeze(0)) ** 2).sum(-1)
        scores = -(mean_term + cov_term)
        if top_k is not None:
            scores, idx = torch.topk(scores, k=min(top_k, scores.size(1)), dim=1)
        else:
            idx = torch.argsort(scores, dim=1, descending=True)
            scores = torch.gather(scores, 1, idx)
        return scores, idx
