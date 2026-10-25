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
    # 视为「活跃」的最少选课数：MOOCCube 用户中位仅 2 门课，阈值 5 时时间序切分后
    # 测试集只剩约 1 门课、监督信号太弱；取 10 时仍有 6,548 人可选。
    mooccube_min_user_courses: int = 10
    # 抽样后再过滤一次：抽样是均匀覆盖的，不保证人人够训练（见语义召回的时间序切分）
    mooccube_min_courses_after_sample: int = 8

    # 文本编码器（语义召回的基础）
    encoder_kind: str = "local"           # local | sentence_transformer
    encoder_dim: int = 384                # 对齐 all-MiniLM-L6-v2
    encoder_cache: bool = True            # 编码结果落盘复用
    encoder_ngram_min: int = 1            # 本地降级实现的字符 n-gram 下界
    encoder_ngram_max: int = 2            # 二元组实测优于三元组（Top-10 类目准确率 0.618 vs 0.605）
    encoder_model: str = "all-MiniLM-L6-v2"   # encoder_kind=sentence_transformer 时使用

    # 阶段一：双塔语义召回
    # deterministic（点表示）| gaussian（分布表示）。
    # 默认 deterministic：高斯扩展在 MOOCCube 上无收益（见 models/recall/gaussian.py 顶部实测结论）。
    tower_kind: str = "deterministic"
    recall_embed_dim: int = 128           # 双塔输出的召回向量维度
    recall_hidden_dim: int = 256          # 塔内隐层宽度
    recall_dropout: float = 0.0
    recall_temperature: float = 0.05      # InfoNCE 温度 τ（沿用双塔常用值）
    recall_lr: float = 1e-3
    recall_epochs: int = 10
    recall_batch_size: int = 256
    recall_neg_items: int = 0             # 每批额外拼入的热门负样本数（0 = 仅 in-batch）
    recall_min_user_train_courses: int = 8  # 用户至少这么多门课才参与训练（否则无历史可推）
    recall_max_need_items: int = 20       # 构造学生需求文本时最多拼接多少门课
    # all_prefixes：每个交互都作正样本（信号最多但慢，5 万样本时约 700s/epoch）
    # last_prefix：每用户只用最后一个交互作正样本（快 ~10 倍，更贴近线上形态）
    recall_sample_mode: str = "last_prefix"
    eval_ks: tuple[int, ...] = (5, 10, 20)
    eval_n_negatives: int = 99            # 采样候选时的负样本数（0 = 全量候选）

    # 阶段二：高斯双塔（概率式表示）
    gaussian_variance_floor: float = 0.05  # 方差正则权重，防止方差塌缩退化成阶段一
    gaussian_max_std: float = 10.0         # 标准差上界，防方差爆炸
    gaussian_history_signal: bool = True   # 把历史长度作为需求塔输入（方差学得出来的前提）
    sparse_history_max: int = 2            # 「历史稀疏」评估：训练期只保留最近 N 条交互
                                           # （用于检验概率建模在冷启动/模糊需求上的价值）
    # 质量信号融合权重。CourseHub 在自建 MOOC 数据上网格搜索得到 (0.7, 0.2, 0.1)；
    # MOOCCube 上的实测不同：语义与热度取 0.5/0.5 时 HitRate@10 = 0.580，
    # 优于纯语义 0.443 与纯热度 0.493（MOOCCube 无评分，rating 权重为 0）。
    # 因此默认值按本数据集实测调整，而不是照抄论文。
    quality_w_semantic: float = 0.6
    quality_w_rating: float = 0.0
    quality_w_popularity: float = 0.4

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
            min_courses_after_sample=self.mooccube_min_courses_after_sample,
            id_map_path=os.path.join(self.model_dir, "mooccube_id_map.json"),
        )
