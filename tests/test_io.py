from engine.data.schema import User, Resource, Behavior, Rating, DataBundle
from engine.data.io import save_bundle, load_bundle


def _make_bundle() -> DataBundle:
    return DataBundle(
        users=[User(user_id=1), User(user_id=2)],
        resources=[
            Resource(resource_id=10, type="course", category_id=2, tags=(), metadata={}),
            Resource(resource_id=11, type="video", category_id=3, tags=("AI", "ML"), metadata={}),
        ],
        behaviors=[
            Behavior(user_id=1, resource_id=10, action="view", ts=0),
            Behavior(user_id=2, resource_id=11, action="click", ts=1),
        ],
        ratings=[
            Rating(user_id=1, resource_id=10, score=5, ts=1),
            Rating(user_id=2, resource_id=11, score=4, ts=2),
        ],
    )


def test_empty_tags_roundtrip(tmp_path):
    b = _make_bundle()
    save_bundle(b, str(tmp_path))
    loaded = load_bundle(str(tmp_path))
    r0 = [r for r in loaded.resources if r.resource_id == 10][0]
    assert r0.tags == ()


def test_normal_tags_roundtrip(tmp_path):
    b = _make_bundle()
    save_bundle(b, str(tmp_path))
    loaded = load_bundle(str(tmp_path))
    r1 = [r for r in loaded.resources if r.resource_id == 11][0]
    assert r1.tags == ("AI", "ML")


def test_behaviors_ratings_counts_preserved(tmp_path):
    b = _make_bundle()
    save_bundle(b, str(tmp_path))
    loaded = load_bundle(str(tmp_path))
    assert len(loaded.users) == 2
    assert len(loaded.resources) == 2
    assert len(loaded.behaviors) == 2
    assert len(loaded.ratings) == 2


def test_description_roundtrip(tmp_path):
    import dataclasses
    b = _make_bundle()
    b = DataBundle(
        users=b.users,
        resources=[
            dataclasses.replace(b.resources[0], description="讲解 Python 基础语法"),
            b.resources[1],
        ],
        behaviors=b.behaviors,
        ratings=b.ratings,
    )
    save_bundle(b, str(tmp_path))
    loaded = load_bundle(str(tmp_path))
    with_desc = [r for r in loaded.resources if r.resource_id == 10][0]
    without_desc = [r for r in loaded.resources if r.resource_id == 11][0]
    assert with_desc.description == "讲解 Python 基础语法"
    assert without_desc.description == ""


def test_legacy_resources_csv_without_description_column(tmp_path):
    """旧版 resources.csv 没有 description 列时，读取应降级为空串而非报错。"""
    import pandas as pd
    b = _make_bundle()
    save_bundle(b, str(tmp_path))
    df = pd.read_csv(tmp_path / "resources.csv").drop(columns=["description"])
    df.to_csv(tmp_path / "resources.csv", index=False)
    loaded = load_bundle(str(tmp_path))
    assert all(r.description == "" for r in loaded.resources)
