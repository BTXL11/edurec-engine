"""不确定性门控的表示融合。

问题背景
--------
序列主干（STOSA 一类）对每个物品给出的不是点向量，而是分布 N(μ, Σ)：
协方差 σ 描述「用户此刻的偏好有多集中」。多模态侧（PRISM 的图文向量）与
知识侧（EMBEDD-ER 的概念向量）提供的是与点击历史互补、但噪声更大的证据。

融合的朴素做法是固定权重 ``v = 0.6·v_id + 0.4·v_mm``；本模块把权重交给
不确定性：

    g(u) = sigmoid(a · z(u) + b),      z(u) = (u - mean) / std
    μ_fused = (1 - g) · μ_id + g · μ_mm
    Σ_fused = (1 - g) · Σ_id + g · Σ_mm

其中 u 是用户侧的不确定性（序列分布协方差的迹）。直觉是：兴趣越分散，
越该借外部证据；兴趣越集中，越该信自己的行为历史。a 由验证集学出，
门控是否成立本身就是一个可证伪的命题（a 不显著为正即假设不成立）。

设计约束
--------
* 只依赖 numpy：脚本、离线评估、单元测试都能直接调用，不引入 torch 依赖。
* 全部输入按「对角协方差」处理（STOSA 的 covariance embedding 即对角形式），
  避免在原型阶段引入完整协方差矩阵的开销与数值风险。
* 打分统一用对角高斯下的 2-Wasserstein 距离，融合前后口径一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-8


def sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def summarize_uncertainty(cov: np.ndarray) -> np.ndarray:
    """对角协方差 → 每个样本一个标量不确定性（各维方差之和，即矩阵的迹）。

    参数
    ----
    cov : (n, d) 或 (d,) 的对角方差。

    返回
    ----
    (n,) 的不确定性；单样本输入返回标量形状 (1,)。
    """
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    return np.clip(cov, 0.0, None).sum(axis=1)


def wasserstein2(
    mean_a: np.ndarray,
    var_a: np.ndarray,
    mean_b: np.ndarray,
    var_b: np.ndarray,
) -> np.ndarray:
    """两个对角高斯之间 2-Wasserstein 距离的平方。

    ``W2² = ||μ_a − μ_b||² + ||σ_a − σ_b||²``，其中 σ 为标准差。
    两项同量纲（都在嵌入空间里），因此无需额外加权即可相加。
    """
    mean_a = np.asarray(mean_a, dtype=np.float64)
    mean_b = np.asarray(mean_b, dtype=np.float64)
    std_a = np.sqrt(np.clip(np.asarray(var_a, dtype=np.float64), _EPS, None))
    std_b = np.sqrt(np.clip(np.asarray(var_b, dtype=np.float64), _EPS, None))
    mean_term = ((mean_a - mean_b) ** 2).sum(axis=-1)
    std_term = ((std_a - std_b) ** 2).sum(axis=-1)
    return mean_term + std_term


def fuse_gaussian(
    mean_id: np.ndarray,
    var_id: np.ndarray,
    mean_mm: np.ndarray,
    var_mm: np.ndarray,
    weight: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """按门控权重凸组合两路高斯表示。

    ``weight`` 是「多模态侧」的权重 g：0 表示纯 ID，1 表示纯多模态。
    形状可以是标量、``(n,)`` 或 ``(n, 1)``，自动广播到 ``(n, d)``。
    """
    mean_id = np.asarray(mean_id, dtype=np.float64)
    var_id = np.asarray(var_id, dtype=np.float64)
    mean_mm = np.asarray(mean_mm, dtype=np.float64)
    var_mm = np.asarray(var_mm, dtype=np.float64)

    g = np.asarray(weight, dtype=np.float64)
    if g.ndim == 1:
        g = g[:, None]
    if g.ndim == 0:
        g = float(g)  # 标量权重保持标量，避免把 (d,) 拍成 (1, d)
    g = np.clip(g, 0.0, 1.0)

    fused_mean = (1.0 - g) * mean_id + g * mean_mm
    fused_var = np.clip((1.0 - g) * var_id + g * var_mm, _EPS, None)
    return fused_mean, fused_var


@dataclass
class UncertaintyGate:
    """由用户不确定性驱动的两路融合门控。

    拟合参数只有两个（标准化尺度上的斜率 ``slope`` 与截距 ``intercept``），
    在几百个验证用户上即可稳定估计，避免小数据下过拟合。

    ``fit`` 的标签约定：对每个验证用户，如果纯多模态表示把真实下一件物品
    排得比纯 ID 表示更靠前，记 ``label = 1``。斜率显著为正即说明
    「越不确定的用户越该用多模态」，这正是待验证的假设。
    """

    slope: float = 0.0
    intercept: float = 0.0
    mean_unc: float = 0.0
    std_unc: float = 1.0
    n_train: int = 0
    converged: bool = field(default=False)

    # ---------------------------------------------------------------- 变换
    def standardize(self, unc: np.ndarray) -> np.ndarray:
        unc = np.asarray(unc, dtype=np.float64).reshape(-1)
        return (unc - self.mean_unc) / max(self.std_unc, _EPS)

    def logits(self, unc: np.ndarray) -> np.ndarray:
        return self.slope * self.standardize(unc) + self.intercept

    def weights(self, unc: np.ndarray) -> np.ndarray:
        """给定不确定性，返回多模态侧权重 g∈[0,1]，形状 ``(n,)``。"""
        return sigmoid(self.logits(unc))

    # ---------------------------------------------------------------- 拟合
    def fit(
        self,
        unc: np.ndarray,
        label: np.ndarray,
        *,
        l2: float = 1e-2,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> "UncertaintyGate":
        """用牛顿法（IRLS）拟合两参数 logistic 回归。

        退化情形（标签全 0 / 全 1）直接落到常数门控并把 ``converged`` 置 False，
        调用方据此判断「这条数据上根本学不出方向」。
        """
        unc = np.asarray(unc, dtype=np.float64).reshape(-1)
        label = np.asarray(label, dtype=np.float64).reshape(-1)
        if unc.size != label.size:
            raise ValueError(f"unc 与 label 长度不一致：{unc.size} vs {label.size}")
        if unc.size == 0:
            raise ValueError("empty training set")

        self.mean_unc = float(unc.mean())
        self.std_unc = float(unc.std())
        if self.std_unc < _EPS:
            self.std_unc = 1.0

        base = float(np.clip(label.mean(), 1e-6, 1 - 1e-6))
        self.intercept = float(np.log(base / (1 - base)))
        self.slope = 0.0
        self.n_train = int(unc.size)

        if label.min() == label.max():
            self.converged = False
            return self

        z = self.standardize(unc)
        design = np.stack([z, np.ones_like(z)], axis=1)  # (n, 2)
        beta = np.array([self.slope, self.intercept], dtype=np.float64)
        reg = np.diag([float(l2), 0.0])

        for _ in range(max_iter):
            p = sigmoid(design @ beta)
            w = np.clip(p * (1 - p), 1e-8, None)
            grad = design.T @ (label - p) - reg @ beta
            hess = design.T @ (design * w[:, None]) + reg
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:  # pragma: no cover - 数值兜底
                break
            beta = beta + step
            if np.max(np.abs(step)) < tol:
                self.converged = True
                break

        self.slope = float(beta[0])
        self.intercept = float(beta[1])
        return self

    # ------------------------------------------------------------ 便捷接口
    def fuse(
        self,
        mean_id: np.ndarray,
        var_id: np.ndarray,
        mean_mm: np.ndarray,
        var_mm: np.ndarray,
        unc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """先按不确定性算权重，再融合两路高斯。"""
        return fuse_gaussian(mean_id, var_id, mean_mm, var_mm, self.weights(unc))

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "mean_unc": self.mean_unc,
            "std_unc": self.std_unc,
            "n_train": self.n_train,
            "converged": self.converged,
        }
