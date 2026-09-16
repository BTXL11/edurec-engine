import numpy as np
import pytest

from engine.config import EngineConfig
from engine.data.schema import User, Resource, Behavior, DataBundle
from engine.features.encoder import LocalTfidfSvdEncoder
from engine.models.recall.semantic import TwoTowerSemantic
from engine.models.recall.trainer import build_interactions, train_two_tower
from engine.pipeline.recall_pipeline import (
    evaluate_popularity_sampled, evaluate_sampled, evaluate_text_similarity_sampled,
    need_text_from, sample_candidates, split_user_interactions,
)

# 两个语义簇，各 4 门课——每簇至少 4 门，时间序切分后测试期才会有「训练期没见过」的课
_MATH_ITEMS = (10, 20, 30, 40)
_COOK_ITEMS = (110, 120, 130, 140)
_MATH_TEXT = {
    10: "线性代数。矩阵、向量空间与特征值分解。",
    20: "微积分。极限、导数与积分。",
    30: "概率论。随机变量与分布。",
    40: "数理统计。假设检验与回归。",
}
_COOK_TEXT = {
    110: "烹饪基础。刀工、火候与调味。",
    120: "烘焙入门。面团发酵与烤箱温度。",
    130: "家常菜。红烧与清蒸技法。",
    140: "面点制作。包子馒头与面条。",
}
_TEXT = {**_MATH_TEXT, **_COOK_TEXT}
_TITLES = {10: "线性代数", 20: "微积分", 30: "概率论", 40: "数理统计",
           110: "烹饪基础", 120: "烘焙入门", 130: "家常菜", 140: "面点制作"}


def _resources() -> list[Resource]:
    return [
        Resource(resource_id=i, type="course",
                 category_id=0 if i in _MATH_ITEMS else 1, tags=(),
                 metadata={"title": _TITLES[i]}, description=_TEXT[i])
        for i in sorted(_TEXT)
    ]


def _bundle(behavior_rows) -> DataBundle:
    users = sorted({u for u, _, _ in behavior_rows})
    return DataBundle(
        users=[User(user_id=u) for u in users],
        resources=_resources(),
        behaviors=[Behavior(user_id=u, resource_id=r, action="view", ts=t)
                   for u, r, t in behavior_rows],
        ratings=[],
    )


def _history_bundle() -> DataBundle:
    """4 个用户，每人 24 条交互，**排布保证测试期一定出现新课**。

    每人的序列刻意设计成：
        前 16 条只用本簇前两门课  → 切分后必然全部落在训练期
        后  8 条才出现本簇后两门课 → 落在验证/测试期
    于是「正样本 = 测试期课程 − 训练期已选」非空，评估才成立。
    （若让各门课前期均匀出现，训练期就会覆盖全部课程，测试期没有新课可推。）
    """
    rows, ts = [], 1_000
    plan = {1: _MATH_ITEMS, 2: tuple(reversed(_MATH_ITEMS)),
            3: _COOK_ITEMS, 4: tuple(reversed(_COOK_ITEMS))}
    for user, items in plan.items():
        early, late = items[:2], items[2:]
        sequence = list(early) * 8 + list(late) * 4        # 16 + 8 = 24 条
        for item in sequence:
            ts += 1
            rows.append((user, item, ts))
    return _bundle(rows)


def _cfg(**kw) -> EngineConfig:
    base = dict(seed=0, recall_epochs=2, recall_batch_size=4, recall_embed_dim=16,
                recall_hidden_dim=32, encoder_dim=32, encoder_cache=False,
                eval_ks=(1, 2, 4), eval_n_negatives=3)
    base.update(kw)
    return EngineConfig(**base)


# --- 时间序切分 ---

def test_split_keeps_chronological_order_and_no_overlap():
    split = split_user_interactions(build_interactions(_history_bundle()), 0.5, 0.25)
    for user in split.train:
        tr, va, te = split.train[user], split.val.get(user, []), split.test.get(user, [])
        assert tr and va and te                       # 三段都有内容
        assert max(x.ts for x in tr) < min(x.ts for x in va)
        assert max(x.ts for x in va) < min(x.ts for x in te)


def test_split_leaves_new_items_for_test():
    """测试期应当出现训练期没选过的课，否则评估无从谈起（夹具本身的自检）。"""
    split = split_user_interactions(build_interactions(_history_bundle()), 0.5, 0.25)
    for user, targets in split.test.items():
        seen = {it.item for it in split.train[user]}
        assert {it.item for it in targets} - seen, f"user {user} 测试期没有新课"


def test_split_gives_short_histories_to_train():
    """交互太少的用户整体进训练集，避免出现无法评估的空段。"""
    bundle = _bundle([(1, 10, 1), (1, 20, 2)])
    split = split_user_interactions(build_interactions(bundle))
    assert 1 in split.train
    assert not split.val.get(1) and not split.test.get(1)


def test_need_text_uses_recent_history_only():
    bundle = _bundle([(1, 10, 1), (1, 20, 2), (1, 30, 3)])
    ints = build_interactions(bundle)
    text = need_text_from(ints[:2], max_items=1)
    assert text == ints[1].text                        # 只取最近 1 条
    assert need_text_from([], max_items=5) == ""


# --- 候选采样 ---

def test_sample_candidates_contains_positives_and_excludes_seen():
    item_ids = [10, 20, 30, 40]
    rng = np.random.default_rng(0)
    cands = sample_candidates(rng, user=1, positives={20}, seen={10},
                              item_ids=item_ids, n_negatives=10)
    assert 20 in cands                                  # 正样本必在
    assert 10 not in cands                              # 已交互的必不在
    assert set(cands) <= set(item_ids)


def test_sample_candidates_returns_all_when_pool_small():
    item_ids = [10, 20, 30]
    rng = np.random.default_rng(0)
    cands = sample_candidates(rng, user=1, positives={10}, seen=set(),
                              item_ids=item_ids, n_negatives=99)
    assert cands == item_ids


def test_sample_candidates_is_reproducible():
    item_ids = list(range(100))
    a = sample_candidates(np.random.default_rng(1), 1, {5}, {7}, item_ids, 10)
    b = sample_candidates(np.random.default_rng(1), 1, {5}, {7}, item_ids, 10)
    assert a == b


# --- 评估协议 ---

def _trained(cfg=None):
    cfg = cfg or _cfg()
    bundle = _history_bundle()
    split = split_user_interactions(build_interactions(bundle), 0.5, 0.25)
    train_ints = [it for acts in split.train.values() for it in acts]
    item_ids = sorted(r.resource_id for r in bundle.resources)
    enc = LocalTfidfSvdEncoder(dim=cfg.encoder_dim, seed=cfg.seed)
    model = train_two_tower(enc, train_ints, item_ids, cfg)
    return model, enc, bundle, split, item_ids, cfg


def test_evaluate_sampled_reports_random_baseline():
    model, enc, bundle, split, item_ids, cfg = _trained()
    metrics = evaluate_sampled(model, enc, bundle, split, item_ids, cfg,
                               n_negatives=cfg.eval_n_negatives)
    assert metrics["n_users"] > 0
    assert 0.0 <= metrics["hitrate@1"] <= 1.0
    assert metrics["mean_candidates"] >= 1
    assert metrics["random_hitrate@1"] == pytest.approx(1 / metrics["mean_candidates"])


def test_evaluate_sampled_is_reproducible():
    model, enc, bundle, split, item_ids, cfg = _trained()
    a = evaluate_sampled(model, enc, bundle, split, item_ids, cfg, n_negatives=3)
    b = evaluate_sampled(model, enc, bundle, split, item_ids, cfg, n_negatives=3)
    assert a == b                                   # 采样种子固定 → 候选一致


def test_evaluate_sampled_respects_n_negatives():
    model, enc, bundle, split, item_ids, cfg = _trained()
    small = evaluate_sampled(model, enc, bundle, split, item_ids, cfg, n_negatives=1)
    large = evaluate_sampled(model, enc, bundle, split, item_ids, cfg, n_negatives=5)
    assert small["mean_candidates"] < large["mean_candidates"]


def test_all_baselines_use_same_candidate_protocol():
    """三种评估必须产出同一批用户/候选规模，否则横向比较不成立。"""
    model, enc, bundle, split, item_ids, cfg = _trained()
    n = cfg.eval_n_negatives
    two_tower = evaluate_sampled(model, enc, bundle, split, item_ids, cfg, n_negatives=n)
    text = evaluate_text_similarity_sampled(enc, bundle, split, item_ids, cfg, n_negatives=n)
    pop = evaluate_popularity_sampled(split, item_ids, cfg, n_negatives=n)
    assert two_tower["n_users"] > 0
    for m in (text, pop):
        assert m["n_users"] == two_tower["n_users"]
        assert m["mean_candidates"] == pytest.approx(two_tower["mean_candidates"])


def test_trained_model_beats_pure_text_similarity_on_synthetic_data():
    """合成数据里语义簇分明：训练后的双塔应不弱于未训练的纯文本相似度。"""
    model, enc, bundle, split, item_ids, cfg = _trained()
    n = cfg.eval_n_negatives
    two_tower = evaluate_sampled(model, enc, bundle, split, item_ids, cfg, n_negatives=n)
    text = evaluate_text_similarity_sampled(enc, bundle, split, item_ids, cfg, n_negatives=n)
    assert two_tower["hitrate@2"] >= text["hitrate@2"]


def test_evaluate_sampled_handles_no_evaluable_users():
    bundle = _bundle([(1, 10, 1)])
    split = split_user_interactions(build_interactions(bundle))
    model = TwoTowerSemantic(in_dim=8, out_dim=4, hidden=6)
    enc = LocalTfidfSvdEncoder(dim=8, seed=0).fit([_MATH_TEXT[10]])
    assert evaluate_sampled(model, enc, bundle, split, sorted(_TEXT), _cfg()) == {}
