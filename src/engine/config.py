from __future__ import annotations
import os
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

    # MOOCCube（data_source=mooccube 时使用）
    mooccube_dir: str = "dataset/MOOCCube/MOOCCube"
    mooccube_max_users: int = 5000        # 活跃用户抽样上限（0 = 不限）
    mooccube_min_user_courses: int = 5    # 视为「活跃」的最少选课数

    # 文本编码器（语义召回的基础）
    encoder_kind: str = "local"           # local | sentence_transformer
    encoder_dim: int = 384                # 对齐 all-MiniLM-L6-v2
    encoder_cache: bool = True            # 编码结果落盘复用
    encoder_ngram_min: int = 1            # 本地降级实现的字符 n-gram 下界
    encoder_ngram_max: int = 2            # 二元组实测优于三元组（Top-10 类目准确率 0.618 vs 0.605）
    encoder_model: str = "all-MiniLM-L6-v2"   # encoder_kind=sentence_transformer 时使用

    @classmethod
    def from_yaml(cls, path: str) -> "EngineConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, allow_unicode=True)

    def mooccube_config(self):
        """按当前配置构造 MOOCCubeConfig（延迟 import，避免循环依赖）。"""
        from .data.mooccube import MOOCCubeConfig
        return MOOCCubeConfig(
            root=self.mooccube_dir,
            max_users=self.mooccube_max_users,
            min_user_courses=self.mooccube_min_user_courses,
            id_map_path=os.path.join(self.model_dir, "mooccube_id_map.json"),
        )
