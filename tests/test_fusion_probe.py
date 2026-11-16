"""融合探针的采样与分数级融合：不依赖真实数据集的单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 允许 ``python -m pytest tests`` 找不到 scripts 包
    sys.path.insert(0, str(ROOT))

from scripts import run_fusion_probe as probe  # noqa: E402


def test_build_candidates_stays_inside_pool() -> None:
    rng = np.random.default_rng(0)
    pool = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    history = np.array([1, 2, 3, 10], dtype=np.int64)

    negs = probe.build_candidates(rng, n_items=100, history=history, neg=20, pool=pool)

    assert negs.shape == (20,)
    assert set(np.unique(negs)).issubset(set(pool.tolist()))
    assert 10 not in negs  # 历史里的物品不能再当负样本


def test_build_candidates_without_pool_avoids_history() -> None:
    rng = np.random.default_rng(1)
    history = np.array([1, 2, 3, 4, 5], dtype=np.int64)

    negs = probe.build_candidates(rng, n_items=50, history=history, neg=30)

    assert negs.shape == (30,)
    assert np.unique(negs).size == 30
    assert negs.min() >= 1 and negs.max() <= 50
    assert not set(negs.tolist()) & set(history.tolist())


def test_rank_fusion_is_deterministic_and_bounded() -> None:
    rng = np.random.default_rng(2)
    n_items, dim, neg = 40, 8, 9
    mean_id = rng.normal(size=(n_items + 1, dim))
    mean_id[0] = 0.0
    var_id = np.ones_like(mean_id) + 0.1 * rng.random((n_items + 1, dim))
    feats = rng.normal(size=(n_items + 1, 2 * dim))
    feats[0] = 0.0
    users = [np.array([1, 2, 3, 4]), np.array([5, 6, 7]), np.array([8, 9, 10, 11, 12])]

    weight_fns = {"id": None, "half": lambda t, d: np.full_like(t, 0.5)}

    first = probe.evaluate_rank_fusion(
        users, mean_id, var_id, feats, k=3, neg=neg, seed=11, weight_fns=weight_fns
    )
    second = probe.evaluate_rank_fusion(
        users, mean_id, var_id, feats, k=3, neg=neg, seed=11, weight_fns=weight_fns
    )

    for key in weight_fns:
        np.testing.assert_array_equal(first[0][key], second[0][key])
        assert first[0][key].min() >= 1
        assert first[0][key].max() <= neg + 1
    np.testing.assert_allclose(first[1], second[1])


def test_rank_fusion_pool_only_ranks_cold_candidates() -> None:
    rng = np.random.default_rng(3)
    n_items, dim, neg = 30, 6, 5
    mean_id = rng.normal(size=(n_items + 1, dim))
    mean_id[0] = 0.0
    var_id = np.ones_like(mean_id)
    feats = rng.normal(size=(n_items + 1, dim))
    feats[0] = 0.0
    users = [np.array([1, 2, 3, 20])]
    pool = np.array([20, 21, 22], dtype=np.int64)

    ranks, _, _ = probe.evaluate_rank_fusion(
        users,
        mean_id,
        var_id,
        feats,
        k=2,
        neg=neg,
        seed=5,
        weight_fns={"id": None},
        pool=pool,
    )

    # 目标 20 在池内，候选数 = neg + 1；名次只可能落在 1..neg+1。
    assert ranks["id"][0] <= neg + 1


def test_ridge_map_cross_fitting_shapes_and_effect() -> None:
    rng = np.random.default_rng(4)
    n_items, n_feat, n_dim = 40, 10, 5
    feats = rng.normal(size=(n_items + 1, n_feat))
    feats[0] = 0.0
    target = rng.normal(size=(n_items + 1, n_dim))
    target[0] = 0.0
    train_ids = np.arange(1, n_items + 1)

    pred_plain, var_plain = probe.fit_ridge_map(feats, target, train_ids, folds=0)
    pred_oof, var_oof = probe.fit_ridge_map(feats, target, train_ids, folds=5)

    assert pred_plain.shape == target.shape
    assert var_plain.shape == target.shape
    assert np.isfinite(pred_oof).all()
    assert (var_oof > 0).all()
    # 交叉拟合让训练物品的预测不再等于样本内解，两者必须有差异。
    assert not np.allclose(pred_plain[train_ids], pred_oof[train_ids])


def test_dispersion_of_single_item_history_is_zero() -> None:
    mean = np.arange(1, 7, dtype=np.float64).reshape(3, 2)
    assert probe.interest_dispersion(np.array([2]), mean, k=5) == 0.0
