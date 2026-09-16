"""离线评估指标：命中类（召回）与区分度类（排序）。

一期没有评分预测任务，因此不包含 RMSE；指标口径对齐 CourseHub 的 MRR / HitRate@K / NDCG@K，
外加召回常用的 Recall@K。
"""

from __future__ import annotations

import numpy as np


def recall_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    """召回率：相关项里有多少落在前 K。"""
    if not relevant:
        return 0.0
    hit = sum(1 for x in ranked[:k] if x in relevant)
    return hit / len(relevant)


def hit_rate_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    """前 K 是否命中至少一个相关项（0/1）。"""
    return 1.0 if any(x in relevant for x in ranked[:k]) else 0.0


def ndcg_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    """位置加权的命中质量：越靠前命中得分越高，按理想排序归一。"""
    dcg = 0.0
    for pos, x in enumerate(ranked[:k], start=1):
        if x in relevant:
            dcg += 1.0 / np.log2(pos + 1)
    ideal = sum(1.0 / np.log2(pos + 1)
                for pos in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def mean_reciprocal_rank(relevant: set[int], ranked: list[int]) -> float:
    """第一个命中位置的倒数；未命中记 0。"""
    for pos, x in enumerate(ranked, start=1):
        if x in relevant:
            return 1.0 / pos
    return 0.0


def aggregate_ranking_metrics(relevant_of: dict[int, set[int]],
                              ranked_of: dict[int, list[int]],
                              ks: tuple[int, ...] = (5, 10, 20)) -> dict[str, float]:
    """对多个用户聚合命中类指标；只统计在 `relevant_of` 里且有排序结果的用户。"""
    users = [u for u in relevant_of if relevant_of[u] and u in ranked_of]
    if not users:
        return {}
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(
            [recall_at_k(relevant_of[u], ranked_of[u], k) for u in users]))
        out[f"hitrate@{k}"] = float(np.mean(
            [hit_rate_at_k(relevant_of[u], ranked_of[u], k) for u in users]))
        out[f"ndcg@{k}"] = float(np.mean(
            [ndcg_at_k(relevant_of[u], ranked_of[u], k) for u in users]))
    out["mrr"] = float(np.mean(
        [mean_reciprocal_rank(relevant_of[u], ranked_of[u]) for u in users]))
    out["n_users"] = float(len(users))
    return out


def auc_from_scores(y_true, y_score) -> float:
    """二分类 AUC（Mann-Whitney U，**并列分数取平均秩**）；正负样本缺失时返回 0.5。

    必须处理并列：`argsort` 不是稳定排序，若直接按名次累加，全并列的正样本会被
    排到未尾而得到 AUC=1.0 —— 这是严重高估。平均秩给出正确的 0.5。
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(scores)
    rank_sum = float(ranks[y_true == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    """1 基的平均秩：并列值共享它们所占名次的平均值。"""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0            # 名次 i+1 … j+1 的平均
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks
