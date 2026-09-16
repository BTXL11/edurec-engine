import numpy as np
import pytest
import torch

from engine.config import EngineConfig
from engine.models.recall.gaussian import GaussianTwoTower
from engine.models.recall.trainer import build_model


def _model(**kw) -> GaussianTwoTower:
    torch.manual_seed(0)
    base = dict(in_dim=8, out_dim=4, hidden=6)
    base.update(kw)
    return GaussianTwoTower(**base)


# --- 分布参数 ---

def test_distributions_have_expected_shapes_and_unit_means():
    m = _model()
    x = torch.randn(5, 8)
    mu, std = m.student_dist(x)
    assert mu.shape == (5, 4) and std.shape == (5, 4)
    assert torch.allclose(mu.norm(dim=-1), torch.ones(5), atol=1e-5)   # 单位球面
    assert (std > 0).all()                                            # 方差必为正


def test_std_is_clamped():
    m = _model(min_std=0.01, max_std=2.0)
    _, std = m.item_dist(torch.randn(64, 8) * 50)
    assert float(std.detach().min()) >= 0.01 - 1e-6
    assert float(std.detach().max()) <= 2.0 + 1e-6


def test_two_towers_are_independent():
    m = _model()
    x = torch.randn(4, 8)
    mu_s, std_s = m.student_dist(x)
    mu_i, std_i = m.item_dist(x)
    assert not torch.allclose(mu_s, mu_i)


# --- Wasserstein 距离 ---

def test_wasserstein_is_zero_for_identical_distributions():
    m = _model()
    mu, std = m.item_dist(torch.randn(3, 8))
    d = m.wasserstein2(mu, std, mu, std)
    assert torch.allclose(d, torch.zeros(3), atol=1e-6)


def test_wasserstein_decomposes_into_mean_and_variance_terms():
    """W₂² = ‖Δμ‖² + Σ_d (Δσ)²——逐项手算核对。"""
    mu_a = torch.tensor([[1.0, 0.0]])
    std_a = torch.tensor([[0.5, 0.5]])
    mu_b = torch.tensor([[0.0, 2.0]])
    std_b = torch.tensor([[1.5, 0.0]])
    d = GaussianTwoTower.wasserstein2(mu_a, std_a, mu_b, std_b)
    mean_term = (1.0 ** 2 + 2.0 ** 2)                 # 5.0
    var_term = ((0.5 - 1.5) ** 2 + (0.5 - 0.0) ** 2)  # 1.25
    assert float(d[0]) == pytest.approx(mean_term + var_term)


def test_same_mean_but_different_variance_is_not_zero_distance():
    """这正是点表示做不到的：均值相同、确定性不同，距离应当非零。"""
    mu = torch.tensor([[1.0, 0.0]])
    close = GaussianTwoTower.wasserstein2(
        mu, torch.tensor([[0.1, 0.1]]), mu, torch.tensor([[0.1, 0.1]]))
    far = GaussianTwoTower.wasserstein2(
        mu, torch.tensor([[0.1, 0.1]]), mu, torch.tensor([[2.0, 2.0]]))
    assert float(close[0]) == pytest.approx(0.0)
    assert float(far[0]) > 0


def test_score_is_negative_distance():
    m = _model()
    need, res = torch.randn(4, 8), torch.randn(4, 8)
    mu_u, std_u = m.student_dist(need)
    mu_i, std_i = m.item_dist(res)
    expect = -m.wasserstein2(mu_u, std_u, mu_i, std_i)
    assert torch.allclose(m.score(need, res), expect, atol=1e-6)


# --- U-GLAD 式方差校准 ---

def test_calibrated_mean_suppresses_when_uncertain():
    m = _model()
    mu = torch.tensor([[1.0, 1.0]])
    certain = m.calibrated_mean(mu, torch.tensor([[0.1, 0.1]]))
    uncertain = m.calibrated_mean(mu, torch.tensor([[2.0, 2.0]]))
    assert float(uncertain.norm()) < float(certain.norm())
    # 方差为 0 时不压低
    assert torch.allclose(m.calibrated_mean(mu, torch.zeros(1, 2)), mu)


def test_calibrated_mean_never_flips_sign():
    m = _model()
    mu = torch.tensor([[1.0, -1.0]])
    out = m.calibrated_mean(mu, torch.tensor([[100.0, 100.0]]))
    assert torch.allclose(out, torch.zeros(1, 2))      # clamp 到 0，不反向


# --- 训练与接口 ---

def test_infonce_loss_decreases():
    torch.manual_seed(0)
    m = _model()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    need = torch.randn(8, 8)
    pos = need + 0.01 * torch.randn(8, 8)
    first = float(m(need, pos).detach())
    for _ in range(20):
        loss = m(need, pos)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(m(need, pos).detach()) < first


def test_variance_regularizer_penalizes_collapse():
    """方差趋近 0 时惩罚项应当更大，从而阻止塌缩退化成阶段一。"""
    tiny = GaussianTwoTower.variance_regularizer(
        torch.full((1, 4), 1e-3), torch.full((1, 4), 1e-3))
    normal = GaussianTwoTower.variance_regularizer(
        torch.full((1, 4), 1.0), torch.full((1, 4), 1.0))
    wide = GaussianTwoTower.variance_regularizer(
        torch.full((1, 4), 3.0), torch.full((1, 4), 3.0))
    assert float(tiny) > float(normal) > float(wide)
    assert float(normal) == pytest.approx(0.0)     # σ=1 → 惩罚为 0


def test_variance_floor_changes_loss():
    m0 = _model(variance_floor=0.0)
    m1 = _model(variance_floor=1.0)
    need, pos = torch.randn(4, 8), torch.randn(4, 8)
    assert float(m0(need, pos).detach()) != float(m1(need, pos).detach())


def test_encode_items_returns_mean_and_std():
    m = _model()
    mu, std = m.encode_items(torch.randn(10, 8))
    assert mu.shape == (10, 4) and std.shape == (10, 4)
    assert (std > 0).all()


def test_encode_items_handles_empty():
    m = _model()
    mu, std = m.encode_items(torch.zeros(0, 8))
    assert mu.shape == (0, 4) and std.shape == (0, 4)


def test_rank_uses_wasserstein_and_sorts_descending():
    m = _model()
    stored = m.encode_items(torch.randn(10, 8))
    mu_i, std_i = stored
    query = torch.arange(8, dtype=torch.float32)
    scores, idx = m.rank(query, stored, history_len=torch.tensor([3]))
    assert scores.shape == (1, 10) and idx.shape == (1, 10)
    assert torch.all(scores[0, :-1] >= scores[0, 1:] - 1e-6)
    mu_u, std_u = m.student_dist(query.unsqueeze(0), torch.tensor([3]))
    ref = -(m.wasserstein2(mu_u.expand(10, -1), std_u.expand(10, -1), mu_i, std_i))
    assert torch.allclose(scores[0], ref[idx[0]], atol=1e-6)


# --- 工厂 ---

def test_uncertainty_report_flags_uninformative_variance():
    """诊断方法本身要能用：常量方差应被报出「变异系数 ≈ 0」。"""
    m = _model()
    item_emb = torch.randn(32, 8)
    need_emb = torch.randn(32, 8)
    report = m.uncertainty_report(need_emb, torch.arange(1, 33), item_emb)
    assert set(report) >= {"item_var_mean", "item_var_cv", "student_var_mean",
                           "history_corr"}
    assert report["item_var_mean"] > 0
    assert 0.0 <= report["item_var_cv"] < 1.0
    assert -1.0 <= report["history_corr"] <= 1.0


def test_uncertainty_report_detects_perfect_negative_correlation():
    """构造一个「历史越短方差越大」的假模型，诊断应当报出强负相关。"""
    m = _model(history_signal=False)

    class Fake(GaussianTwoTower):
        def student_dist(self, need_emb, history_len=None):
            mu = torch.nn.functional.normalize(need_emb, dim=-1)
            if history_len is None:
                history_len = torch.ones(need_emb.size(0))
            # 历史越短 → 方差越大（反比），这才是「学到不确定性」的样子
            std = (10.0 / history_len.float()).clamp(min=1e-3)
            return mu, std.unsqueeze(-1).expand_as(mu)

    fake = Fake(in_dim=8, out_dim=4, hidden=6)
    report = fake.uncertainty_report(torch.randn(16, 8), torch.arange(1, 17))
    # 反比关系（非线性）下 Pearson 相关约 -0.57；诊断关心的是**方向为负**，
    # 而不是精确的线性强度，因此阈值取 -0.4。
    assert report["history_corr"] < -0.4


def test_uncertainty_report_without_history_length():
    m = _model()
    report = m.uncertainty_report(torch.randn(8, 8), None)
    assert "history_corr" not in report
    assert report["student_var_mean"] > 0


def test_build_model_selects_tower_kind():
    cfg = EngineConfig()
    det = build_model("deterministic", 8, cfg)
    gau = build_model("gaussian", 8, cfg)
    from engine.models.recall.semantic import TwoTowerSemantic
    assert isinstance(det, TwoTowerSemantic)
    assert isinstance(gau, GaussianTwoTower)


def test_build_model_rejects_unknown_kind():
    with pytest.raises(ValueError, match="tower_kind"):
        build_model("magic", 8, EngineConfig())


def test_build_model_respects_config_dims():
    cfg = EngineConfig(recall_embed_dim=32, recall_hidden_dim=16)
    m = build_model("gaussian", 8, cfg)
    assert m.out_dim == 32
    mu, std = m.student_dist(torch.randn(2, 8))
    assert mu.shape == (2, 32) and std.shape == (2, 32)
