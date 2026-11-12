import numpy as np
import pytest

from engine.fusion.gate import (
    UncertaintyGate,
    fuse_gaussian,
    sigmoid,
    summarize_uncertainty,
    wasserstein2,
)


def _make_pair(n=64, seed=0):
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=n)
    var = rng.uniform(0.1, 2.0, size=n)
    return mean, var


# ------------------------------------------------------------------ 基础算子
def test_sigmoid_is_stable_at_extremes():
    out = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0)


def test_summarize_uncertainty_is_trace_of_diagonal():
    cov = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 4.0]])
    np.testing.assert_allclose(summarize_uncertainty(cov), [6.0, 4.0])
    np.testing.assert_allclose(summarize_uncertainty(cov[0]), [6.0])


def test_wasserstein2_is_zero_for_identical_gaussians():
    mean, var = _make_pair()
    assert wasserstein2(mean, var, mean, var) == pytest.approx(0.0, abs=1e-12)


def test_wasserstein2_is_symmetric_and_positive_otherwise():
    mean_a, var_a = _make_pair(seed=1)
    mean_b, var_b = _make_pair(seed=2)
    ab = wasserstein2(mean_a, var_a, mean_b, var_b)
    ba = wasserstein2(mean_b, var_b, mean_a, var_a)
    assert ab == pytest.approx(ba)
    assert ab > 0


def test_wasserstein2_decomposes_into_mean_and_std_terms():
    """距离平方应恰好等于均值项 + 标准差项，便于论文里解释量纲。"""
    mean_a, var_a = _make_pair(seed=3)
    mean_b, var_b = _make_pair(seed=4)
    expected = ((mean_a - mean_b) ** 2).sum() + (
        (np.sqrt(var_a) - np.sqrt(var_b)) ** 2
    ).sum()
    assert wasserstein2(mean_a, var_a, mean_b, var_b) == pytest.approx(expected)


# ------------------------------------------------------------------ 融合
def test_fuse_gaussian_endpoints_select_single_source():
    mean_id, var_id = _make_pair(seed=5)
    mean_mm, var_mm = _make_pair(seed=6)
    m0, v0 = fuse_gaussian(mean_id, var_id, mean_mm, var_mm, 0.0)
    np.testing.assert_allclose(m0, mean_id)
    np.testing.assert_allclose(v0, var_id)
    m1, v1 = fuse_gaussian(mean_id, var_id, mean_mm, var_mm, 1.0)
    np.testing.assert_allclose(m1, mean_mm)
    np.testing.assert_allclose(v1, var_mm)


def test_fuse_gaussian_keeps_variance_positive_and_broadcasts_per_row():
    n, d = 4, 8
    mean_id = np.zeros((n, d))
    var_id = np.ones((n, d))
    mean_mm = np.ones((n, d))
    var_mm = np.full((n, d), 4.0)
    g = np.array([0.0, 0.25, 0.75, 1.0])[:, None]
    mean, var = fuse_gaussian(mean_id, var_id, mean_mm, var_mm, g)
    assert mean.shape == (n, d)
    assert np.all(var > 0)
    np.testing.assert_allclose(mean[:, 0], [0.0, 0.25, 0.75, 1.0])
    np.testing.assert_allclose(var[:, 0], [1.0, 1.75, 3.25, 4.0])


# ------------------------------------------------------------------ 门控
def test_gate_weights_are_bounded_and_monotone():
    gate = UncertaintyGate().fit(
        np.linspace(0.1, 2.0, 200),
        (np.linspace(0.1, 2.0, 200) > 1.0).astype(float),
    )
    unc = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    w = gate.weights(unc)
    assert np.all((w >= 0.0) & (w <= 1.0))
    assert np.all(np.diff(w) > 0)


def test_gate_learns_positive_slope_when_multimodal_wins_on_uncertain_users():
    """构造：不确定性越高，多模态越该赢 → 斜率必须显著为正（假设成立）。"""
    rng = np.random.default_rng(0)
    unc = rng.uniform(0.0, 1.0, 4000)
    prob = 0.1 + 0.8 * unc  # 越不确定，多模态赢的概率越高
    label = (rng.uniform(size=unc.size) < prob).astype(float)
    gate = UncertaintyGate().fit(unc, label)
    assert gate.converged
    assert gate.slope > 0.5


def test_gate_learns_negative_slope_when_signal_is_reversed():
    rng = np.random.default_rng(1)
    unc = rng.uniform(0.0, 1.0, 4000)
    prob = 0.9 - 0.8 * unc
    label = (rng.uniform(size=unc.size) < prob).astype(float)
    gate = UncertaintyGate().fit(unc, label)
    assert gate.slope < -0.5


def test_gate_falls_back_to_constant_on_degenerate_labels():
    gate = UncertaintyGate().fit(np.linspace(0, 1, 100), np.zeros(100))
    assert gate.converged is False
    w = gate.weights(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(w, w[0], atol=1e-12)
    assert 0.0 < w[0] < 1.0


def test_gate_fuse_matches_manual_fuse():
    gate = UncertaintyGate().fit(
        np.linspace(0.1, 2.0, 200), (np.linspace(0.1, 2.0, 200) > 1.0).astype(float)
    )
    n, d = 3, 6
    mean_id, var_id = np.zeros((n, d)), np.ones((n, d))
    mean_mm, var_mm = np.ones((n, d)), np.full((n, d), 2.0)
    unc = np.array([0.2, 1.0, 1.8])
    m, v = gate.fuse(mean_id, var_id, mean_mm, var_mm, unc)
    g = gate.weights(unc)[:, None]
    np.testing.assert_allclose(m, g * mean_mm + (1 - g) * mean_id)
    np.testing.assert_allclose(v, g * var_mm + (1 - g) * var_id)


def test_gate_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        UncertaintyGate().fit(np.zeros(3), np.zeros(4))

# ------------------------------------------------------------------ 配对检验
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_fusion_probe.py"


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("run_fusion_probe", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_stats_is_null_when_rankings_identical():
    probe = _load_probe_module()
    ranks = np.array([1, 2, 5, 11, 30, 99], dtype=np.int64)
    stat = probe.paired_stats(ranks, ranks, n_boot=50, seed=0)
    assert stat["delta"] == pytest.approx(0.0)
    assert stat["wins"] == 0 and stat["losses"] == 0
    assert stat["p_value"] == pytest.approx(1.0)


def test_paired_stats_detects_a_real_improvement():
    probe = _load_probe_module()
    base = np.full(400, 40, dtype=np.int64)          # 全部未命中
    new = np.where(np.arange(400) < 200, 3, 40)      # 一半改命中
    stat = probe.paired_stats(base, new, n_boot=50, seed=0)
    assert stat["delta"] == pytest.approx(0.5)
    assert stat["wins"] == 200 and stat["losses"] == 0
    assert stat["p_value"] < 1e-6
    assert stat["ci95"][0] > 0.4


def test_interest_dispersion_is_zero_for_a_stable_history():
    probe = _load_probe_module()
    mean = np.zeros((6, 8), dtype=np.float64)
    seq = np.array([3, 3, 3, 3], dtype=np.int64)
    assert probe.interest_dispersion(seq, mean, k=5) == pytest.approx(0.0)
