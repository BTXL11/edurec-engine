"""阶段一评估流水线：时间序切分 → 训练 → 在所有资源上排名 → 命中类指标。

评估协议（**防泄漏的关键**，两条都不能少）：

  1. **需求只用训练期交互构造**。如果需求里含有该用户测试期的课程，
     模型等于把答案抄进了输入；
  2. **候选里屏蔽训练期已交互的课程**。否则「已经选过的课」会因为 ID 记忆
     天然得高分，指标虚高到没有意义。

因此每个用户的评估是：用训练期历史构造需求 → 给全量资源打分 →
剔除他训练期已选过的 → 看测试期课程排在第几。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features.encoder import resource_text
from ..models.recall.trainer import (
    PREFIX_SEP, Interaction, prepare_item_matrix, score_users,
)
from .evaluate import aggregate_ranking_metrics


@dataclass
class UserSequenceSplit:
    """按时间序切好的用户交互序列。"""

    train: dict[int, list[Interaction]]
    val: dict[int, list[Interaction]]
    test: dict[int, list[Interaction]]

    def all_users(self) -> set[int]:
        return set(self.train) | set(self.val) | set(self.test)


def split_user_interactions(interactions: list[Interaction],
                            train_ratio: float = 0.8,
                            val_ratio: float = 0.1) -> UserSequenceSplit:
    """按每个用户自己的时间序切分（80/10/10），不做全局时间切分。

    全局切分的问题：不同用户的时间跨度差异很大，统一时间点会让某些用户整段落入
    测试集、另一些整段落入训练集。按用户切分保证人人都有训练历史可推。
    """
    by_user: dict[int, list[Interaction]] = {}
    for it in interactions:
        by_user.setdefault(it.user, []).append(it)

    train: dict[int, list[Interaction]] = {}
    val: dict[int, list[Interaction]] = {}
    test: dict[int, list[Interaction]] = {}
    for user, acts in by_user.items():
        acts = sorted(acts, key=lambda x: (x.ts, x.item))
        n = len(acts)
        n_tr = int(n * train_ratio)
        n_va = int(n * (train_ratio + val_ratio))
        # 三段都要有内容才有评估意义；历史太短的用户整体留给训练
        if n < 3:
            train[user] = acts
            continue
        n_tr = max(1, min(n_tr, n - 2))
        n_va = max(n_tr + 1, min(n_va, n - 1))
        train[user] = acts[:n_tr]
        val[user] = acts[n_tr:n_va]
        test[user] = acts[n_va:]
    return UserSequenceSplit(train=train, val=val, test=test)


def need_text_from(history: list[Interaction], max_items: int = 20) -> str:
    """用一段历史构造学生需求文本（取最近的 max_items 条）。"""
    acts = sorted(history, key=lambda x: (x.ts, x.item))
    return PREFIX_SEP.join(it.text for it in acts[-max_items:] if it.text)


def evaluate_ranking(model, encoder, bundle, split: UserSequenceSplit,
                     item_ids: list[int], config,
                     use_val: bool = False) -> dict[str, float]:
    """在验证/测试集上评估排名质量。

    指标按 CourseHub 的口径：MRR / HitRate@K / NDCG@K，另加 Recall@K。
    """
    item_texts = [next((resource_text(r) for r in bundle.resources
                        if r.resource_id == i), "") for i in item_ids]
    prepare_item_matrix(model, encoder, item_texts)

    target_split = split.val if use_val else split.test
    index_of = {item: i for i, item in enumerate(item_ids)}

    need_texts: list[str] = []
    users: list[int] = []
    for user, targets in target_split.items():
        if not targets:
            continue
        history = split.train.get(user, [])
        if not history:
            continue          # 没有训练历史 → 属于冷启动，另行统计
        need_texts.append(need_text_from(history, config.recall_max_need_items))
        users.append(user)

    if not users:
        return {}

    scores = score_users(model, encoder, need_texts)          # (U, n_items)

    # 屏蔽训练期已交互的课程
    seen_of: dict[int, set[int]] = {}
    for user in users:
        seen_of[user] = {index_of[it.item] for it in split.train.get(user, [])
                         if it.item in index_of}

    relevant_of: dict[int, set[int]] = {}
    ranked_of: dict[int, list[int]] = {}
    for row, user in enumerate(users):
        mask = np.ones(len(item_ids), dtype=bool)
        seen = seen_of[user]
        if seen:
            mask[list(seen)] = False
        masked = np.where(mask, scores[row], -np.inf)
        ranked_of[user] = np.argsort(-masked).tolist()
        relevant_of[user] = {index_of[it.item] for it in target_split[user]
                             if it.item in index_of}

    ks = getattr(config, "eval_ks", (5, 10, 20))
    metrics = aggregate_ranking_metrics(relevant_of, ranked_of, ks=tuple(ks))
    metrics["cold_start_users"] = float(
        sum(1 for u in target_split if target_split[u] and not split.train.get(u)))
    return metrics


def popularity_baseline(split: UserSequenceSplit, item_ids: list[int],
                        config) -> dict[str, float]:
    """热门兜底基线：不训练、不看需求，直接给最常被选的课排在前面。

    语义模型必须显著超过这条线才有意义——否则「语义」二字站不住。
    """
    from collections import Counter

    counts = Counter(it.item for acts in split.train.values() for it in acts)
    order = [i for i, _ in counts.most_common()]
    order += [i for i in item_ids if i not in set(order)]
    ranked = [item_ids.index(i) for i in order]

    relevant_of: dict[int, set[int]] = {}
    ranked_of: dict[int, list[int]] = {}
    index_of = {item: i for i, item in enumerate(item_ids)}
    for user, targets in split.test.items():
        if not targets or not split.train.get(user):
            continue
        relevant_of[user] = {index_of[it.item] for it in targets if it.item in index_of}
        ranked_of[user] = ranked
    ks = getattr(config, "eval_ks", (5, 10, 20))
    return aggregate_ranking_metrics(relevant_of, ranked_of, ks=tuple(ks))


# --- 1 vs N 采样候选协议 ---

def sample_candidates(rng: np.random.Generator, user: int, positives: set[int],
                      seen: set[int], item_ids: list[int], n_negatives: int) -> list[int]:
    """为用户抽一批候选：全部正样本 + N 个「他未交互过」的负样本。

    全量候选（几百上千门课）下正样本占比极低，指标既难解释也难以区分模型；
    采样候选是推荐系统的标准做法——随机排序的期望值恰好是 `命中数 / 候选数`，
    于是「强于随机多少倍」成为可读的参照。
    """
    pool = [i for i in item_ids if i not in seen and i not in positives]
    if n_negatives <= 0 or len(pool) <= n_negatives:
        negatives = pool
    else:
        negatives = [pool[k] for k in rng.choice(len(pool), size=n_negatives, replace=False)]
    return sorted(positives | set(negatives))


def _sampled_frames(split: UserSequenceSplit, item_ids: list[int], config,
                    n_negatives: int, seed: int | None = None,
                    use_val: bool = False):
    """构造采样候选评估用的公共数据：(用户, 候选 ID, 正样本 ID, 需求文本)。

    三种评估（热门基线 / 纯文本相似度 / 训练后双塔）共用同一批候选集，
    否则「谁更强」会被候选差异污染。
    """
    index_of = {item: i for i, item in enumerate(item_ids)}
    rng = np.random.default_rng(config.seed if seed is None else seed)
    target_split = split.val if use_val else split.test
    frames = []
    for user, targets in target_split.items():
        history = split.train.get(user, [])
        if not history or not targets:
            continue
        seen = {it.item for it in history}
        positives = {it.item for it in targets} - seen
        if not positives:
            continue
        candidates = sample_candidates(rng, user, positives, seen, item_ids, n_negatives)
        frames.append({
            "row": len(frames),                    # 与打分矩阵的行号对齐
            "user": user,
            "candidates": candidates,              # 原始 item_id，顺序即索引基准
            "positive_idx": {index_of[p] for p in positives},
            "need": need_text_from(history, config.recall_max_need_items),
            "seen": seen,
        })
    return frames


def _metrics_from_scores(frames, scores_for, item_ids: list[int], config) -> dict:
    """把「每个用户候选上的分数」汇总成命中类指标。

    `scores_for(frame, cand_idx) -> 分数数组`，必须返回**与 cand_idx 同序**的分数。
    """
    index_of = {item: i for i, item in enumerate(item_ids)}
    relevant_of: dict[int, set[int]] = {}
    ranked_of: dict[int, list[int]] = {}
    n_candidates: list[int] = []
    for frame in frames:
        user = frame["user"]
        cand_idx = [index_of[c] for c in frame["candidates"]]
        scores = np.asarray(scores_for(frame, cand_idx), dtype=float)
        order = np.argsort(-scores)
        ranked_of[user] = [cand_idx[o] for o in order]
        relevant_of[user] = frame["positive_idx"]
        n_candidates.append(len(cand_idx))
    ks = tuple(getattr(config, "eval_ks", (5, 10, 20)))
    metrics = aggregate_ranking_metrics(relevant_of, ranked_of, ks=ks)
    if metrics and n_candidates:
        mean_cand = float(np.mean(n_candidates))
        metrics["mean_candidates"] = mean_cand
        for k in ks:
            metrics[f"random_hitrate@{k}"] = min(1.0, k / mean_cand)
    return metrics


def evaluate_sampled(model, encoder, bundle, split: UserSequenceSplit,
                     item_ids: list[int], config,
                     n_negatives: int = 99, seed: int | None = None,
                     use_val: bool = False) -> dict[str, float]:
    """采样候选下的评估：每个用户在自己的候选集里排序，统计正样本的名次。

    候选集 = 该用户测试期选过的课 + `n_negatives` 门他没选过的课。
    随机排序的期望 HitRate@K ≈ K / |候选|，因此报告里同时给出随机基线。
    """
    item_texts = [next((resource_text(r) for r in bundle.resources
                        if r.resource_id == i), "") for i in item_ids]
    prepare_item_matrix(model, encoder, item_texts)
    frames = _sampled_frames(split, item_ids, config, n_negatives, seed, use_val)
    if not frames:
        return {}
    need_texts = [f["need"] for f in frames]
    score_matrix = score_users(model, encoder, need_texts)     # (U, n_items)

    def scores_for(frame, cand_idx):
        return score_matrix[frame["row"]][cand_idx]

    return _metrics_from_scores(frames, scores_for, item_ids, config)


def evaluate_popularity_sampled(split: UserSequenceSplit, item_ids: list[int], config,
                                n_negatives: int = 99, seed: int | None = None,
                                use_val: bool = False) -> dict[str, float]:
    """采样候选下的热门基线：只按训练期被选次数排序，不看需求。"""
    from collections import Counter

    counts = Counter(it.item for acts in split.train.values() for it in acts)
    frames = _sampled_frames(split, item_ids, config, n_negatives, seed, use_val)
    if not frames:
        return {}

    def scores_for(frame, cand_idx):
        return np.array([counts.get(item_ids[i], 0) for i in cand_idx], dtype=float)

    return _metrics_from_scores(frames, scores_for, item_ids, config)


def evaluate_text_similarity_sampled(encoder, bundle, split: UserSequenceSplit,
                                     item_ids: list[int], config,
                                     n_negatives: int = 99, seed: int | None = None,
                                     use_val: bool = False) -> dict[str, float]:
    """纯文本相似度基线（**不训练**）：需求文本编码与资源编码直接做余弦。

    这条线是「双塔训练到底有没有用」的判据：
    若训练后的双塔打不过它，说明可学习的投影没有带来额外收益，
    方案就该退回到 CourseHub 那种「编码即表示」的形态。
    """
    item_texts = [next((resource_text(r) for r in bundle.resources
                        if r.resource_id == i), "") for i in item_ids]
    encoder.fit(item_texts)
    item_mat = encoder.encode(item_texts)                       # (n_items, D)
    frames = _sampled_frames(split, item_ids, config, n_negatives, seed, use_val)
    if not frames:
        return {}
    need_mat = encoder.encode([f["need"] for f in frames])      # (U, D)
    sim = need_mat @ item_mat.T

    def scores_for(frame, cand_idx):
        return sim[frame["row"]][cand_idx]

    return _metrics_from_scores(frames, scores_for, item_ids, config)
