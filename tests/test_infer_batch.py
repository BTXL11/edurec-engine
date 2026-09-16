"""批量推理与交接契约的测试。

重点钉住 `docs/platform-contract.md` 的硬约束——这些约束一旦破了，
platform 侧会整份导入失败（HTTP 500），因此必须有测试守住。
"""

import json

import numpy as np
import pytest

from engine.config import EngineConfig
from engine.data.schema import User, Resource, Behavior, Rating, DataBundle
from engine.features.encoder import LocalTfidfSvdEncoder
from engine.models.recall.trainer import build_interactions, train_two_tower
from engine.pipeline.infer_batch import (
    BatchResult, Recommendation, infer_batch, write_handoff,
)
from engine.pipeline.recall_pipeline import split_user_interactions

_MATH = "线性代数。矩阵、向量空间与特征值分解。"
_MATH2 = "微积分。极限、导数与积分。"
_COOK = "烹饪基础。刀工、火候与调味。"
_COOK2 = "烘焙入门。面团发酵与烤箱温度。"


def _bundle(rows, ratings=()) -> DataBundle:
    resources = [
        Resource(resource_id=10, type="course", category_id=0, tags=(),
                 metadata={"title": "线性代数"}, description=_MATH),
        Resource(resource_id=20, type="course", category_id=0, tags=(),
                 metadata={"title": "微积分"}, description=_MATH2),
        Resource(resource_id=30, type="course", category_id=1, tags=(),
                 metadata={"title": "烹饪基础"}, description=_COOK),
        Resource(resource_id=40, type="course", category_id=1, tags=(),
                 metadata={"title": "烘焙入门"}, description=_COOK2),
    ]
    users = sorted({u for u, _, _ in rows} | {9})       # 用户 9 无行为 = 冷启动
    return DataBundle(
        users=[User(user_id=u) for u in users],
        resources=resources,
        behaviors=[Behavior(user_id=u, resource_id=r, action="view", ts=t)
                   for u, r, t in rows],
        ratings=[Rating(user_id=u, resource_id=r, score=s, ts=t)
                 for u, r, s, t in ratings],
    )


def _history_rows():
    rows, ts = [], 1_000
    for user, items in {1: (10, 20, 30, 40), 2: (20, 10, 40, 30)}.items():
        for rep in range(3):
            for item in items:
                ts += 1
                rows.append((user, item, ts))
    return rows


def _cfg(**kw) -> EngineConfig:
    base = dict(seed=0, recall_epochs=2, recall_batch_size=4, recall_embed_dim=16,
                recall_hidden_dim=32, encoder_dim=32, encoder_cache=False,
                top_n=2, quality_w_semantic=0.6, quality_w_rating=0.0,
                quality_w_popularity=0.4)
    base.update(kw)
    return EngineConfig(**base)


def _trained(cfg=None, bundle=None):
    cfg = cfg or _cfg()
    bundle = bundle or _bundle(_history_rows())
    split = split_user_interactions(build_interactions(bundle), 0.8, 0.1)
    train_ints = [it for acts in split.train.values() for it in acts]
    item_ids = sorted(r.resource_id for r in bundle.resources)
    enc = LocalTfidfSvdEncoder(dim=cfg.encoder_dim, seed=cfg.seed)
    model = train_two_tower(enc, train_ints, item_ids, cfg)
    return model, enc, bundle, item_ids, split, cfg


# --- 批量推理语义 ---

def test_infer_covers_all_users_including_cold_start():
    model, enc, bundle, item_ids, split, cfg = _trained()
    result = infer_batch(model, enc, bundle, cfg, item_ids=item_ids)
    assert set(result.recommendations) == {u.user_id for u in bundle.users}
    assert result.cold_start_users == 1                    # 用户 9 没有行为
    cold = result.recommendations[9]
    assert len(cold) > 0                                   # 冷启动也有兜底推荐


def test_infer_respects_top_n_and_deduplicates():
    model, enc, bundle, item_ids, split, cfg = _trained()
    result = infer_batch(model, enc, bundle, cfg, item_ids=item_ids)
    for rec in result.recommendations.values():
        assert len(rec.item_ids) <= cfg.top_n
        assert len(set(rec.item_ids)) == len(rec.item_ids)   # 无重复
        assert len(rec.scores) == len(rec.item_ids)          # 分数一一对应


def test_infer_excludes_already_seen_items():
    """契约要求推荐里不含用户已交互过的课程。"""
    model, enc, bundle, item_ids, split, cfg = _trained()
    result = infer_batch(model, enc, bundle, cfg, item_ids=item_ids)
    seen = {}
    for b in bundle.behaviors:
        seen.setdefault(b.user_id, set()).add(b.resource_id)
    for user, rec in result.recommendations.items():
        assert not (set(rec.item_ids) & seen.get(user, set()))


def test_infer_is_reproducible():
    model, enc, bundle, item_ids, split, cfg = _trained()
    a = infer_batch(model, enc, bundle, cfg, item_ids=item_ids).as_id_lists()
    b = infer_batch(model, enc, bundle, cfg, item_ids=item_ids).as_id_lists()
    assert a == b


def test_infer_target_users_subset():
    model, enc, bundle, item_ids, split, cfg = _trained()
    result = infer_batch(model, enc, bundle, cfg, item_ids=item_ids, target_users=[1])
    assert set(result.recommendations) == {1}


def test_infer_uses_full_history_by_default_and_train_only_on_request():
    """两种历史口径都要满足契约，且「已交互屏蔽」按各自口径生效。"""
    model, enc, bundle, item_ids, split, cfg = _trained()
    full = infer_batch(model, enc, bundle, cfg, item_ids=item_ids)
    train_only = infer_batch(model, enc, bundle, cfg, item_ids=item_ids,
                             split=split, use_train_only=True)
    seen_all: dict[int, set[int]] = {}
    seen_train: dict[int, set[int]] = {}
    for b in bundle.behaviors:
        seen_all.setdefault(b.user_id, set()).add(b.resource_id)
    for user, acts in split.train.items():
        seen_train.setdefault(user, set()).update(it.item for it in acts)

    for rec in full.recommendations.values():
        assert len(rec.item_ids) <= cfg.top_n
        assert not (set(rec.item_ids) & seen_all.get(rec.user_id, set()))
    for rec in train_only.recommendations.values():
        assert len(rec.item_ids) <= cfg.top_n
        assert not (set(rec.item_ids) & seen_train.get(rec.user_id, set()))

    # 全量口径：用户 9 在 bundle 里没有任何行为 → 冷启动走热门兜底
    assert full.cold_start_users == 1
    assert len(full.recommendations[9]) > 0
    # 训练期口径：用户 9 的历史不足 3 条，切分时被整体留给训练集而不在 split.train，
    # 因此它同样没有可用历史。这说明「历史口径」是由传入的 split 决定的。
    assert train_only.cold_start_users == 1


def test_infer_uses_ratings_when_available():
    """有评分时质量融合的评分项参与排序（MOOCCube 无评分，这条覆盖快照场景）。"""
    rows = _history_rows()
    ratings = [(1, 40, 5, 2000), (2, 30, 5, 2001)]
    bundle = _bundle(rows, ratings)
    cfg = _cfg(quality_w_rating=0.5, quality_w_semantic=0.3, quality_w_popularity=0.2)
    model, enc, bundle, item_ids, split, cfg = _trained(cfg, bundle)
    result = infer_batch(model, enc, bundle, cfg, item_ids=item_ids)
    assert all(len(r.item_ids) <= cfg.top_n for r in result.recommendations.values())


def test_infer_handles_bundle_without_behaviors():
    bundle = _bundle([])
    cfg = _cfg()
    enc = LocalTfidfSvdEncoder(dim=cfg.encoder_dim, seed=cfg.seed).fit([_MATH])
    from engine.models.recall.semantic import TwoTowerSemantic
    model = TwoTowerSemantic(in_dim=cfg.encoder_dim, out_dim=cfg.recall_embed_dim,
                             hidden=cfg.recall_hidden_dim)
    result = infer_batch(model, enc, bundle, cfg, item_ids=sorted(
        r.resource_id for r in bundle.resources))
    assert set(result.recommendations) == {u.user_id for u in bundle.users}
    assert result.cold_start_users == len(bundle.users)


# --- 交接文件契约 ---

def _write(tmp_path, result, **kw):
    main = tmp_path / "recommendations.json"
    env = tmp_path / "recommendations.meta.json"
    envelope = {"contract_version": 1, "model": {"name": "t", "version": "v1"}}
    write_handoff(result, str(main), str(env), envelope, **kw)
    return main, env


def test_main_file_shape_matches_contract(tmp_path):
    result = BatchResult(recommendations={
        1: Recommendation(1, [42, 17, 305], [0.9, 0.8, 0.7]),
        5: Recommendation(5, [3], [0.5]),
    })
    main, _ = _write(tmp_path, result)
    raw = main.read_text(encoding="utf-8")
    assert not raw.startswith("\ufeff")                    # 无 BOM
    payload = json.loads(raw)
    assert payload == {"1": [42, 17, 305], "5": [3]}
    assert all(isinstance(k, str) and k.isdigit() for k in payload)
    assert all(isinstance(v, list) for v in payload.values())
    assert all(isinstance(x, int) and x >= 0
               for v in payload.values() for x in v)


def test_main_file_has_no_extra_top_level_fields(tmp_path):
    """顶层只能有 user_id → 数组，多一个兄弟字段就会让 platform 报 500。"""
    result = BatchResult(recommendations={1: Recommendation(1, [10], [0.1])})
    main, _ = _write(tmp_path, result)
    payload = json.loads(main.read_text(encoding="utf-8"))
    assert set(payload) == {"1"}
    assert not any(k in payload for k in
                   ("contract_version", "generated_at", "model", "top_n"))


def test_write_is_atomic_and_leaves_no_tmp(tmp_path):
    result = BatchResult(recommendations={1: Recommendation(1, [10], [0.1])})
    main, env = _write(tmp_path, result)
    assert main.is_file() and env.is_file()
    assert not (tmp_path / "recommendations.json.tmp").exists()
    assert not (tmp_path / "recommendations.meta.json.tmp").exists()


def test_envelope_carries_scores_and_is_separate_file(tmp_path):
    result = BatchResult(recommendations={
        2: Recommendation(2, [11, 7], [0.93, 0.87]),
    })
    main, env = _write(tmp_path, result)
    envelope = json.loads(env.read_text(encoding="utf-8"))
    assert envelope["contract_version"] == 1
    assert envelope["scores"]["2"] == [0.93, 0.87]
    assert envelope["model"]["name"] == "t"
    # 分数顺序必须与主文件数组一一对应
    assert len(envelope["scores"]["2"]) == len(json.loads(
        main.read_text(encoding="utf-8"))["2"])


def test_envelope_is_optional(tmp_path):
    result = BatchResult(recommendations={1: Recommendation(1, [10], [0.1])})
    main = tmp_path / "recommendations.json"
    write_handoff(result, str(main), None, None)
    assert main.is_file()
    assert not (tmp_path / "recommendations.meta.json").exists()


def test_empty_recommendation_lists_are_allowed(tmp_path):
    """空数组表达「本次不更新该用户」，是合法取值，但数量应能自查。"""
    result = BatchResult(recommendations={7: Recommendation(7, [], [])})
    main, _ = _write(tmp_path, result)
    assert json.loads(main.read_text(encoding="utf-8")) == {"7": []}


def test_scores_are_plain_floats(tmp_path):
    result = BatchResult(recommendations={
        1: Recommendation(1, [10], [np.float32(0.25)]),
    })
    _, env = _write(tmp_path, result)
    envelope = json.loads(env.read_text(encoding="utf-8"))
    assert isinstance(envelope["scores"]["1"][0], float)
