"""STOSA 编码器接入的冒烟测试（缺 checkpoint 时整体跳过）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ENGINE_ROOT = Path(__file__).resolve().parents[1]
CKPT = (
    ENGINE_ROOT
    / "STOSA/outputs/DistSAModel-Beauty-64-1-4-gelu-0.0-0.3-100-0.001-0.0-10-1.0-0.005.pt"
)
STOSA_DIR = ENGINE_ROOT / "STOSA"

pytestmark = pytest.mark.skipif(
    not CKPT.exists(), reason="需要 STOSA/outputs 下的复现 checkpoint"
)


@pytest.fixture(scope="module")
def backbone():
    from engine.fusion.stosa_backbone import load_backbone

    return load_backbone(CKPT, STOSA_DIR)


def test_item_gaussians_follow_upstream_transform(backbone):
    """物品方差必须是 elu(cov)+1，与 trainers.dist_predict_full 一致。"""
    mean, var = backbone.item_gaussians()
    assert mean.shape == (12103, 64)
    assert var.shape == mean.shape
    assert var.min() >= 1.0 - 1e-9

    raw = (
        backbone.model.item_cov_embeddings.weight.detach().cpu().numpy().astype(np.float64)
    )
    np.testing.assert_allclose(var, np.maximum(raw, 0.0) + 1.0, rtol=1e-9, atol=1e-9)


def test_encode_returns_finite_gaussians(backbone):
    seqs = [
        np.array([1, 2, 3, 4, 5], dtype=np.int64),
        np.array([7, 9, 11], dtype=np.int64),
        np.array([13] * 60, dtype=np.int64),
    ]
    mean, cov = backbone.encode(seqs)
    assert mean.shape == (3, 64)
    assert cov.shape == (3, 64)
    assert np.isfinite(mean).all()
    assert np.isfinite(cov).all()


def test_encode_is_deterministic_and_cached(backbone):
    seq = np.array([3, 5, 8, 13, 21], dtype=np.int64)
    first_mean, first_cov = backbone.encode([seq])
    second_mean, second_cov = backbone.encode([seq])
    np.testing.assert_array_equal(first_mean, second_mean)
    np.testing.assert_array_equal(first_cov, second_cov)

    # 换一个批次组成，同一段历史的编码不应改变（缓存键是序列内容）
    mixed_mean, _ = backbone.encode([np.array([1, 2, 3]), seq])
    np.testing.assert_array_equal(mixed_mean[1], first_mean[0])
