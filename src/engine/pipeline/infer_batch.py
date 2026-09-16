"""批量推理：全量用户走语义召回 → 质量融合 → 去重 → top-N。

对齐 `docs/platform-contract.md` 的「方向 B」：

  - 主文件必须是 `{平台 user_id: [平台 resource_id, …]}`，按相关性降序、已去重；
  - **覆盖全量用户**，包括平台上没有行为的冷启动用户（走热门兜底）；
  - **必须自行去重**：平台不去重，重复 ID 会原样进缓存；
  - 推荐结果里不含该用户已交互过的课程（`behaviors.csv` 提供了这份信息，不做就浪费了）；
  - 元信息（分数、模型标识、快照 run_id）放进**旁挂信封**，主文件顶层不得加任何字段。

分数融合沿用 CourseHub 的复合打分，权重按本数据集实测调整（见 `EngineConfig`）：
MOOCCube 上语义 0.6 + 热度 0.4 时 HitRate@10 = 0.580，优于纯语义 0.443 与纯热度 0.493。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..features.encoder import resource_text
from ..models.recall.trainer import prepare_item_matrix, score_users


@dataclass
class Recommendation:
    """一个用户的推荐结果：ID 与分数一一对应、顺序即排名。"""

    user_id: int
    item_ids: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.item_ids)


@dataclass
class BatchResult:
    """全量用户的批量推理结果。"""

    recommendations: dict[int, Recommendation] = field(default_factory=dict)
    cold_start_users: int = 0            # 无历史、走热门兜底的用户数
    n_items: int = 0

    def as_id_lists(self) -> dict[int, list[int]]:
        return {u: r.item_ids for u, r in self.recommendations.items()}


def _split_history(behaviors, split, use_train_only: bool):
    """按用户收集历史交互（可选只用训练期的那部分）。

    推理时「用已知的全部历史」是合理的（要预测的是未来），但**评估时**必须只用训练期历史，
    否则会用到测试期信息。两种口径通过 `use_train_only` 区分。
    """
    history: dict[int, list] = {}
    if use_train_only and split is not None:
        for user, acts in split.train.items():
            history[user] = list(acts)
        return history
    for b in behaviors:
        history.setdefault(b.user_id, []).append(b)
    return history


def _as_rows(acts) -> list[tuple[int, int]]:
    """把历史交互归一成 `(resource_id, ts)`。

    两种来源的字段名不同：`DataBundle.behaviors` 是 `Behavior.resource_id`，
    而切分后的 `Interaction` 用 `.item`（命名对齐「item 侧」）。
    在这里适配一次，调用方不必关心自己传的是哪种。
    """
    out = []
    for a in acts:
        rid = getattr(a, "resource_id", None)
        if rid is None:
            rid = getattr(a, "item", None)
        if rid is not None:
            out.append((int(rid), int(getattr(a, "ts", 0))))
    return out


def infer_batch(model, encoder, bundle, config, item_ids: list[int] | None = None,
                split=None, target_users: list[int] | None = None,
                use_train_only: bool = False) -> BatchResult:
    """为每个目标用户产出 top-N。

    `target_users` 默认取 `bundle.users` 的全部用户（对应「覆盖全量用户」的契约要求）；
    冷启动（无历史）用户直接给热门兜底。
    """
    item_ids = item_ids or sorted(r.resource_id for r in bundle.resources)
    index_of = {item: i for i, item in enumerate(item_ids)}

    # 资源文本按 item_ids 顺序编码；顺序必须与后续打分矩阵的列一一对应
    text_of = {r.resource_id: resource_text(r) for r in bundle.resources}
    item_texts = [text_of.get(i, "") for i in item_ids]
    prepare_item_matrix(model, encoder, item_texts)

    # 热度：被交互次数（全量口径，用于质量融合与冷启动兜底）
    from collections import Counter
    counts = Counter(b.resource_id for b in bundle.behaviors)
    popularity = np.array([counts.get(i, 0) for i in item_ids], dtype=np.float32)

    # 评分：快照有 ratings 时才启用（MOOCCube 没有评分，该项自然退化为 0）
    rating_sum: dict[int, list[int]] = {}
    for r in bundle.ratings:
        rating_sum.setdefault(r.resource_id, []).append(r.score)
    ratings = np.array(
        [float(np.mean(rating_sum[i])) if rating_sum.get(i) else 0.0 for i in item_ids],
        dtype=np.float32)

    from ..models.recall.semantic import (
        fuse_quality, normalize_minmax, rank_percentile)
    quality = fuse_quality(
        np.zeros(len(item_ids), dtype=np.float32),      # 语义项稍后逐用户计算
        normalize_minmax(ratings),
        rank_percentile(popularity),
        w_semantic=0.0,                                 # 这里只取「非语义」部分
        w_rating=config.quality_w_rating,
        w_popularity=config.quality_w_popularity)
    quality_total = config.quality_w_semantic + config.quality_w_rating \
        + config.quality_w_popularity

    # 冷启动兜底顺序：热度降序
    popular_order = [item_ids[i] for i in np.argsort(-popularity)]

    history = _split_history(bundle.behaviors, split, use_train_only)
    targets = sorted(target_users) if target_users is not None else sorted(
        u.user_id for u in bundle.users)

    # 只给「有历史」的用户构造需求文本，批量编码一次
    active = [u for u in targets if history.get(u)]
    need_texts = []
    for user in active:
        acts = sorted(_as_rows(history[user]), key=lambda x: (x[1], x[0]))
        recent = acts[-config.recall_max_need_items:]
        need_texts.append(" ".join(text_of.get(rid, "") for rid, _ in recent
                                   if text_of.get(rid)))
    need_users = active
    semantic = (score_users(model, encoder, need_texts)
                if need_texts else np.zeros((0, len(item_ids)), dtype=np.float32))

    # 语义分逐用户 min-max 到 [0,1]，才能与质量分加权（W₂/内积的绝对尺度因模型而异）
    result = BatchResult(n_items=len(item_ids))
    for row, user in enumerate(need_users):
        scores = semantic[row].astype(np.float64)
        lo, hi = float(scores.min()), float(scores.max())
        norm = (scores - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(scores)
        fused = (config.quality_w_semantic * norm + quality) / quality_total

        seen = {rid for rid, _ in _as_rows(history[user])}
        order = np.argsort(-fused)
        picked_ids, picked_scores = [], []
        for idx in order:
            item = item_ids[int(idx)]
            if item in seen:                      # 契约要求：不在推荐里包含已交互内容
                continue
            picked_ids.append(item)
            picked_scores.append(float(fused[int(idx)]))
            if len(picked_ids) >= config.top_n:
                break
        result.recommendations[user] = Recommendation(user, picked_ids, picked_scores)

    # 冷启动用户：热门兜底
    for user in targets:
        if history.get(user):
            continue
        picked = [i for i in popular_order][:config.top_n]
        result.recommendations[user] = Recommendation(
            user, picked, [float(popularity[index_of[i]]) for i in picked])
        result.cold_start_users += 1

    return result


def write_handoff(result: BatchResult, out_path: str, envelope_path: str | None = None,
                  envelope: dict | None = None) -> None:
    """写主文件（+ 可选旁挂信封）。

    契约硬约束（逐条对应 `docs/platform-contract.md`）：

      * 顶层**只有** `{字符串键: 非负整数数组}`，不得添加任何兄弟字段；
      * 键是十进制无符号整数；元素必须是 JSON **整数**（不能是字符串或小数）；
      * 必须无 BOM 的 UTF-8；
      * **先写临时文件再原子重命名**，避免 platform 读到写了一半的文件。
    """
    payload = {str(r.user_id): [int(i) for i in r.item_ids]
               for r in result.recommendations.values()}

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:          # utf-8，不是 utf-8-sig
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, out_path)                            # 原子替换

    if envelope_path and envelope is not None:
        data = dict(envelope)
        data["scores"] = {str(r.user_id): [round(float(s), 6) for s in r.scores]
                          for r in result.recommendations.values() if r.scores}
        parent = os.path.dirname(envelope_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = envelope_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, envelope_path)
