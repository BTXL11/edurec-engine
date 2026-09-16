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
