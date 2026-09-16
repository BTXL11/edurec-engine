import numpy as np
import pytest

from engine.pipeline.evaluate import (
    aggregate_ranking_metrics, auc_from_scores, hit_rate_at_k,
    mean_reciprocal_rank, ndcg_at_k, recall_at_k,
)


def test_recall_at_k():
    assert recall_at_k({1, 3}, [0, 1, 2, 3, 4], k=3) == 0.5      # 命中 1 个 / 共 2 个
    assert recall_at_k({1, 3}, [1, 3, 0, 2, 4], k=5) == 1.0
    assert recall_at_k(set(), [1, 2], k=2) == 0.0                # 无相关项
    assert recall_at_k({9}, [1, 2], k=2) == 0.0


def test_hit_rate_at_k():
    assert hit_rate_at_k({3}, [0, 1, 3], k=3) == 1.0
    assert hit_rate_at_k({3}, [0, 1, 3], k=2) == 0.0
    assert hit_rate_at_k({1, 2}, [5, 6], k=2) == 0.0


def test_ndcg_prefers_hits_at_top():
    relevant = {1, 3}
    better = ndcg_at_k(relevant, [1, 3, 0, 2, 4], k=5)
    worse = ndcg_at_k(relevant, [0, 2, 4, 1, 3], k=5)
    assert better > worse
    assert better == pytest.approx(1.0)          # 理想排序
    assert ndcg_at_k(set(), [1, 2], k=2) == 0.0


def test_ndcg_is_bounded():
    for ranked in ([4, 3, 2, 1], [1, 2, 3, 4], [2, 4, 1, 3]):
        value = ndcg_at_k({1, 3}, ranked, k=4)
        assert 0.0 <= value <= 1.0 + 1e-9


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank({3}, [0, 1, 3]) == pytest.approx(1 / 3)
    assert mean_reciprocal_rank({0}, [0, 1, 2]) == 1.0
    assert mean_reciprocal_rank({9}, [0, 1, 2]) == 0.0


def test_aggregate_ranking_metrics():
    relevant = {0: {1, 3}, 1: {2}, 2: {9}}
    ranked = {0: [1, 0, 3, 2], 1: [0, 1, 2, 3], 2: [0, 1, 2, 3]}
    out = aggregate_ranking_metrics(relevant, ranked, ks=(1, 4))
    assert out["n_users"] == 3
    # hitrate@1：用户 0 的首位是 1（命中），用户 1、2 的首位不是相关项
    assert out["hitrate@1"] == pytest.approx(1 / 3)
    # recall@1：用户 0 命中 1/2，用户 1 命中 0/1（首位是 0），用户 2 命中 0
    assert out["recall@1"] == pytest.approx((0.5 + 0.0 + 0.0) / 3)
    # recall@4：用户 0 命中 2/2，用户 1 命中 1/1，用户 2 仍为 0
    assert out["recall@4"] == pytest.approx((1.0 + 1.0 + 0.0) / 3)
    assert out["mrr"] == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)


def test_aggregate_ignores_users_without_ranking():
    out = aggregate_ranking_metrics({1: {5}}, {}, ks=(1,))
    assert out == {}


def test_auc_from_scores():
    assert auc_from_scores([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert auc_from_scores([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)
    # 完全并列：必须给 0.5，而不是把正样本排到末尾得出 1.0
    assert auc_from_scores([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert auc_from_scores([0, 0, 1, 1], [0.3] * 4) == pytest.approx(0.5)
    # 部分并列用平均秩：正样本分 [0.5, 0.5]，负样本 [0.1, 0.9]
    assert auc_from_scores([0, 0, 1, 1], [0.1, 0.9, 0.5, 0.5]) == pytest.approx(0.5)
    # 只有单一类别时无区分度可言
    assert auc_from_scores([1, 1], [0.1, 0.2]) == 0.5
    assert auc_from_scores([], []) == 0.5


def test_auc_handles_numpy_input():
    y = np.array([0, 1, 1, 0])
    s = np.array([0.2, 0.9, 0.7, 0.1])
    assert auc_from_scores(y, s) == pytest.approx(1.0)
