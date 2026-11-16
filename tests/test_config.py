import pytest

from engine.config import EngineConfig

def test_default_config_fields():
    cfg = EngineConfig()
    assert cfg.seed == 42
    assert cfg.data_source == "sim"
    assert cfg.recall_embed_dim == 128

def test_config_yaml_roundtrip(tmp_path):
    cfg = EngineConfig(sim_n_users=123)
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(str(p))
    loaded = EngineConfig.from_yaml(str(p))
    assert loaded.sim_n_users == 123


def test_default_config_is_valid():
    EngineConfig().validate()


def test_validate_rejects_platform_without_snapshot():
    with pytest.raises(ValueError, match="snapshot_dir"):
        EngineConfig(data_source="platform").validate()


def test_validate_rejects_zero_weight_sum():
    with pytest.raises(ValueError, match="权重之和"):
        EngineConfig(quality_w_semantic=0, quality_w_rating=0,
                     quality_w_popularity=0).validate()


def test_validate_rejects_bad_split_ratios():
    with pytest.raises(ValueError, match=r"train_ratio \+ val_ratio"):
        EngineConfig(train_ratio=0.9, val_ratio=0.2).validate()


def test_validate_rejects_unknown_kinds():
    with pytest.raises(ValueError, match="tower_kind"):
        EngineConfig(tower_kind="magic").validate()
