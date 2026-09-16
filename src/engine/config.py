from __future__ import annotations
from dataclasses import dataclass, asdict
import yaml


@dataclass
class EngineConfig:
    seed: int = 42
    data_source: str = "sim"                      # sim | movielens

    # 模拟器
    sim_n_users: int = 2000
    sim_n_resources: int = 500
    sim_n_categories: int = 12
    sim_n_tags: int = 30
    sim_n_interactions: int = 100_000
    sim_description_sentences: int = 3   # 每份模拟资源简介的要点句数

    # 预处理
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    # 召回
    recall_embed_dim: int = 64
    recall_batch_size: int = 1024
    recall_lr: float = 1e-3
    recall_epochs: int = 10
    recall_tau: float = 0.05
    recall_k: int = 50

    # 排序
    rank_batch_size: int = 256
    rank_lr: float = 1e-3
    rank_epochs: int = 5
    rank_neg_per_pos: int = 3
    rank_w_ctr: float = 1.0
    rank_w_cvr: float = 0.5
    rank_w_rating: float = 0.5

    # 重排
    top_n: int = 20
    mmr_lambda: float = 0.5
    cold_age_days: float = 7.0
    cold_weight: float = 1.2

    # 路径
    data_dir: str = "dataset"
    model_dir: str = "model"
    snapshot_dir: str = ""   # platform 快照目录（data_source=platform 时使用）

    @classmethod
    def from_yaml(cls, path: str) -> "EngineConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, allow_unicode=True)
