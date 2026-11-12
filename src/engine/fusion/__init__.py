"""不确定性门控的表示融合。

对应融合路线的第一块拼图：把序列模型输出的分布表示（μ, Σ）与多模态表示
按「用户当前有多不确定」加权合并。公开接口见 :mod:`engine.fusion.gate`。
"""

from engine.fusion.gate import (
    UncertaintyGate,
    fuse_gaussian,
    summarize_uncertainty,
    wasserstein2,
)

__all__ = [
    "UncertaintyGate",
    "fuse_gaussian",
    "summarize_uncertainty",
    "wasserstein2",
]
