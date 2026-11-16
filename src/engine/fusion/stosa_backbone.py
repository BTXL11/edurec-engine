"""把 STOSA 的序列编码器接进融合实验。

这一层只做三件事，不训练、不改上游代码：

1. 加载 STOSA 官方 checkpoint（`DistSAModel`，Beauty 64 维、1 层、4 头、最大长度 100）；
2. 按上游 `trainers.py:576-579` 的口径，把历史**左填充**到最大长度，取最后一个位置的
   均值/协方差输出作为用户高斯 `N(μ_u, Σ_u)`；
3. 按上游 `dist_predict_full` 的口径给出物品高斯：均值取 `item_mean_embeddings`，
   方差取 `elu(item_cov_embeddings) + 1`。

上游用绝对导入（`from modules import ...`），所以这里把 `STOSA/` 临时放进 `sys.path`。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

#: 复现报告的训练超参（`STOSA/REPRODUCTION_REPORT.md:139-141`）。
UPSTREAM_HYPERPARAMS = dict(
    hidden_size=64,
    num_attention_heads=4,
    hidden_act="gelu",
    attention_probs_dropout_prob=0.0,
    hidden_dropout_prob=0.3,
    initializer_range=0.02,
)


def _import_dist_sa_model(stosa_dir: str | Path):
    """导入 vendored 的 `STOSA/seqmodels.py`（它用绝对导入 `modules`）。"""
    path = str(Path(stosa_dir).resolve())
    added = path not in sys.path
    if added:
        sys.path.insert(0, path)
    try:
        import importlib

        seqmodels = importlib.import_module("seqmodels")
    finally:
        if added:
            sys.path.remove(path)
    return seqmodels.DistSAModel


def _infer_layers(state: dict) -> int:
    layers = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("item_encoder.layer.")
    }
    return max(layers) + 1 if layers else 1


@dataclass
class StosaBackbone:
    """已加载的 STOSA 编码器；只暴露推理需要的两个方法。"""

    model: object
    item_size: int
    max_seq_length: int
    num_users: int
    hidden_size: int
    device: str = "cpu"
    _cache: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- 物品侧
    def item_gaussians(self) -> tuple[np.ndarray, np.ndarray]:
        """按上游 `dist_predict_full` 的口径返回 (均值, 方差)。"""
        import torch

        with torch.no_grad():
            mean = self.model.item_mean_embeddings.weight.detach().cpu().numpy()
            cov = self.model.item_cov_embeddings.weight.detach().cpu().numpy()
        var = np.maximum(np.asarray(cov, dtype=np.float64), 0.0)
        # elu(x) + 1：负值压到 1.0，正值加 1
        elu = np.where(var > 0, var, 0.0) + 1.0
        return (
            np.asarray(mean, dtype=np.float64),
            np.asarray(elu, dtype=np.float64),
        )

    # ---------------------------------------------------------------- 用户侧
    def encode(
        self, sequences, batch_size: int = 512
    ) -> tuple[np.ndarray, np.ndarray]:
        """把一批历史序列编码成用户高斯（最后一个位置的输出，左填充）。

        同一段历史会被验证/网格/测试三轮重复请求，所以按序列内容做缓存。
        """
        import torch

        seqs = [np.asarray(s, dtype=np.int64) for s in sequences]
        cache = self._cache
        keys = [s.tobytes() for s in seqs]
        missing = [i for i, key in enumerate(keys) if key not in cache]
        if missing:
            todo = [seqs[i] for i in missing]
            means: list[np.ndarray] = []
            covs: list[np.ndarray] = []
            self.model.eval()
            with torch.no_grad():
                for start in range(0, len(todo), batch_size):
                    block = todo[start : start + batch_size]
                    ids = np.zeros((len(block), self.max_seq_length), dtype=np.int64)
                    for row, seq in enumerate(block):
                        tail = seq[-self.max_seq_length :]
                        ids[row, self.max_seq_length - tail.size :] = tail
                    input_ids = torch.from_numpy(ids).to(self.device)
                    user_ids = torch.zeros(
                        len(block), dtype=torch.long, device=self.device
                    )
                    mean_out, cov_out, _, _ = self.model.finetune(input_ids, user_ids)
                    means.append(mean_out[:, -1, :].cpu().numpy())
                    covs.append(cov_out[:, -1, :].cpu().numpy())
            new_mean = np.concatenate(means)
            new_cov = np.concatenate(covs)
            for row, i in enumerate(missing):
                cache[keys[i]] = (new_mean[row], new_cov[row])

        out = [cache[key] for key in keys]
        return (
            np.asarray(np.stack([o[0] for o in out]), dtype=np.float64),
            np.asarray(np.stack([o[1] for o in out]), dtype=np.float64),
        )


def load_backbone(
    ckpt: str | Path,
    stosa_dir: str | Path,
    device: str = "cpu",
) -> StosaBackbone:
    """加载 STOSA checkpoint，返回可直接编码序列的对象。"""
    import torch

    DistSAModel = _import_dist_sa_model(stosa_dir)
    state = torch.load(str(ckpt), map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint 不是 state_dict：{type(state)!r}")

    item_size, hidden_size = state["item_mean_embeddings.weight"].shape
    max_seq_length = state["position_mean_embeddings.weight"].shape[0]
    num_users = state["user_margins.weight"].shape[0]

    args = SimpleNamespace(
        item_size=int(item_size),
        num_users=int(num_users),
        max_seq_length=int(max_seq_length),
        num_hidden_layers=_infer_layers(state),
        distance_metric="wasserstein",
        pvn_weight=0.1,
        kernel_param=1.0,
        cuda_condition=False,
        **UPSTREAM_HYPERPARAMS,
    )
    model = DistSAModel(args)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return StosaBackbone(
        model=model,
        item_size=int(item_size),
        max_seq_length=int(max_seq_length),
        num_users=int(num_users),
        hidden_size=int(hidden_size),
        device=device,
    )
