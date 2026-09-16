import numpy as np
import pytest
import torch

from engine.models.recall.semantic import (
    TwoTowerSemantic, fuse_quality, normalize_log, normalize_minmax, rank_percentile,
)


def _model(**kw) -> TwoTowerSemantic:
    torch.manual_seed(0)
    return TwoTowerSemantic(in_dim=8, out_dim=4, hidden=6, **kw)


def test_embeddings_are_unit_vectors_when_normalized():
    m = _model(normalize=True)
    x = torch.randn(5, 8)
    assert torch.allclose(m.student_emb(x).norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(m.item_emb(x).norm(dim=-1), torch.ones(5), atol=1e-5)


def test_unnormalized_mode_keeps_raw_scale():
    m = _model(normalize=False)
    x = torch.randn(3, 8)
    assert not torch.allclose(m.item_emb(x).norm(dim=-1), torch.ones(3))


def test_towers_are_independent():
    """两个塔参数不同，同一输入不应给出相同向量（否则退化成单塔）。"""
    m = _model()
    x = torch.randn(4, 8)
    assert not torch.allclose(m.student_emb(x), m.item_emb(x))


def test_score_matches_manual_dot_product():
    m = _model()
    u, i = torch.randn(3, 8), torch.randn(3, 8)
    expect = (m.student_emb(u) * m.item_emb(i)).sum(-1)
    assert torch.allclose(m.score(u, i), expect, atol=1e-6)


def test_infonce_loss_decreases_with_optimization():
    """可学习性：用固定的一对一映射做几步梯度下降，loss 必须下降。"""
    torch.manual_seed(0)
    m = _model()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    need = torch.randn(8, 8)
    pos = need + 0.01 * torch.randn(8, 8)      # 正样本与需求高度相关
    first = float(m(need, pos).detach())
    for _ in range(20):
        loss = m(need, pos)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(m(need, pos).detach()) < first


def test_infonce_uses_negatives_when_provided():
    m = _model()
    need, pos = torch.randn(4, 8), torch.randn(4, 8)
    without = float(m(need, pos).detach())
    with_neg = float(m(need, pos, neg_emb=torch.randn(6, 8)).detach())
    assert without != with_neg              # 负样本列改变了损失


def test_rank_orders_scores_descending():
    m = _model()
    stored = m.encode_items(torch.randn(10, 8))
    query = torch.arange(8, dtype=torch.float32)
    scores, idx = m.rank(query, stored)
    assert scores.shape == (1, 10) and idx.shape == (1, 10)
    assert torch.all(scores[0, :-1] >= scores[0, 1:] - 1e-6)   # 降序
    assert idx[0].unique().numel() == 10                      # 无重复、未丢失
    # 核心契约：按下标取回的就是排序后的分数（而非另一批数）
    ref = (m.student_emb(query.unsqueeze(0)) @ stored.T)[0]
    assert torch.allclose(scores[0], ref[idx[0]], atol=1e-6)


def test_rank_top_k_limits_output():
    m = _model()
    stored = m.encode_items(torch.randn(10, 8))
    scores, idx = m.rank(torch.randn(2, 8), stored, top_k=3)
    assert scores.shape == (2, 3) and idx.shape == (2, 3)


def test_encode_items_handles_empty():
    m = _model()
    assert m.encode_items(torch.zeros(0, 8)).shape == (0, 4)


# --- 质量信号融合 ---

def test_normalize_minmax():
    out = normalize_minmax(np.array([2.0, 4.0, 6.0], dtype=np.float32))
    assert np.allclose(out, [0.0, 0.5, 1.0])
    assert np.allclose(normalize_minmax(np.array([3.0, 3.0])), [0.0, 0.0])
    assert normalize_minmax(np.array([], dtype=np.float32)).size == 0


def test_normalize_log_compresses_long_tail():
    raw = np.array([0.0, 99.0, 10000.0], dtype=np.float32)
    out = normalize_log(raw)
    assert np.isclose(out.max(), 1.0)
    assert out[0] == 0.0                      # log1p(0)=0：零值保留在最低档
    # 对数压缩后 99 已经接近最大值（线性只有 0.0099）
    assert out[1] > 0.45
    assert out[1] < 1.0


def test_rank_percentile_handles_zero_heavy_counts():
    """零交互占多数时，min-max 会把它们当成「中等热度」；秩百分位把零压在底部。"""
    counts = np.array([0.0, 0.0, 0.0, 5.0, 100.0], dtype=np.float32)
    out = rank_percentile(counts)
    assert np.isclose(out.max(), 1.0)
    assert len(set(out[:3].tolist())) == 1        # 三个零共享同一档
    assert out[2] < out[3] < out[4]               # 单调，且零值远低于最大热度
    assert out[0] < 0.5


def test_rank_percentile_returns_zero_when_no_signal():
    """全零（例如全站热度都是 0）时必须给 0，而不是给每项加一个常数。"""
    assert np.allclose(rank_percentile(np.zeros(5, dtype=np.float32)), 0.0)
    assert np.allclose(rank_percentile(np.array([7.0, 7.0, 7.0])), 0.0)
    assert np.allclose(rank_percentile(np.array([7.0])), 0.0)


def test_rank_percentile_edge_cases():
    assert rank_percentile(np.array([], dtype=np.float32)).size == 0
    assert np.allclose(rank_percentile(np.array([1.0, 2.0])), [0.0, 1.0])


def test_normalize_minmax_supports_lower_is_better():
    raw = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(normalize_minmax(raw, higher_is_better=False), [1.0, 0.5, 0.0])


def test_fuse_quality_default_weights():
    """语义占主导，评分与热度是补充——对齐 CourseHub 的 (0.7, 0.2, 0.1)。"""
    sem = np.array([1.0, 0.0], dtype=np.float32)
    rat = np.array([0.0, 1.0], dtype=np.float32)
    pop = np.array([0.0, 1.0], dtype=np.float32)
    out = fuse_quality(sem, rat, pop)
    assert np.isclose(out[0], 0.7)
    assert np.isclose(out[1], 0.3)


def test_fuse_quality_pure_semantic():
    sem = np.array([0.2, 0.9], dtype=np.float32)
    out = fuse_quality(sem, np.zeros(2), np.zeros(2),
                       w_semantic=1.0, w_rating=0.0, w_popularity=0.0)
    assert np.allclose(out, sem)


def test_fuse_quality_rejects_zero_weights():
    with pytest.raises(ValueError):
        fuse_quality(np.zeros(2), np.zeros(2), np.zeros(2),
                     w_semantic=0.0, w_rating=0.0, w_popularity=0.0)
