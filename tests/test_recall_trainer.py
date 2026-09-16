import numpy as np
import pytest
import torch

from engine.config import EngineConfig
from engine.data.schema import User, Resource, Behavior, DataBundle
from engine.features.encoder import LocalTfidfSvdEncoder
from engine.models.recall.trainer import (
    PREFIX_SEP, Interaction, build_examples, build_interactions, fuse_scores,
    popular_negatives, prepare_item_matrix, score_users, train_two_tower,
)

# 两个语义簇：数学 / 烹饪
_MATH = "线性代数。矩阵、向量空间与特征值分解，数学基础课程。"
_MATH2 = "微积分。极限、导数与积分，数学分析入门。"
_COOK = "烹饪基础。刀工、火候与调味，家常菜制作。"
_COOK2 = "烘焙入门。面团发酵与烤箱温度控制，甜点制作。"


def _resources() -> list[Resource]:
    return [
        Resource(resource_id=10, type="course", category_id=0, tags=(),
                 metadata={"title": "线性代数"}, description=_MATH),
        Resource(resource_id=20, type="course", category_id=0, tags=(),
                 metadata={"title": "微积分"}, description=_MATH2),
        Resource(resource_id=30, type="course", category_id=1, tags=(),
                 metadata={"title": "烹饪基础"}, description=_COOK),
        Resource(resource_id=40, type="course", category_id=1, tags=(),
                 metadata={"title": "烘焙入门"}, description=_COOK2),
    ]


def _bundle(behavior_rows) -> DataBundle:
    users = sorted({u for u, _, _, _ in behavior_rows})
    return DataBundle(
        users=[User(user_id=u) for u in users],
        resources=_resources(),
        behaviors=[Behavior(user_id=u, resource_id=r, action="view", ts=t)
                   for u, r, t, _ in behavior_rows],
        ratings=[],
    )


def _bundle_with_repeat() -> DataBundle:
    """每个用户重复交互同一批资源，让时间序切分后仍有足够正样本。"""
    rows = []
    ts = 1_000
    for u, items in ((1, (10, 20)), (2, (20, 10)), (3, (30, 40)), (4, (40, 30))):
        for rep in range(4):
            for r in items:
                ts += 1
                rows.append((u, r, ts, None))
    return _bundle(rows)


def _cfg(**kw) -> EngineConfig:
    base = dict(seed=0, recall_epochs=3, recall_batch_size=8, recall_embed_dim=16,
                recall_hidden_dim=32, encoder_dim=32, encoder_cache=False)
    base.update(kw)
    return EngineConfig(**base)


# --- 样本构造 ---

def test_build_interactions_attaches_resource_text():
    b = _bundle([(1, 10, 1, None), (1, 30, 2, None)])
    ints = build_interactions(b)
    assert [(i.user, i.item, i.ts) for i in ints] == [(1, 10, 1), (1, 30, 2)]
    assert ints[0].text.startswith("线性代数")
    assert ints[1].text.startswith("烹饪基础")


def test_build_examples_uses_only_history_before_target():
    """防泄漏核心：正样本自身绝不能出现在构造出的需求文本里。"""
    b = _bundle([
        (1, 10, 1, None), (1, 20, 2, None), (1, 30, 3, None),
    ])
    ints = build_interactions(b)
    item_index = {10: 0, 20: 1, 30: 2}
    ex = build_examples(ints, item_index, max_need_items=10)

    assert len(ex) == 2                       # 第 1 条交互没有历史
    assert ex.pos == [1, 2]                   # 预测第 2、3 条
    # 第 1 个样本的需求 = 第 1 条交互的文本；不含第 2 条（待预测）的文本
    assert ex.need_texts[0] == ints[0].text
    assert _MATH2 not in ex.need_texts[0]
    # 第 2 个样本的需求 = 前两条；不含第 3 条
    assert _MATH in ex.need_texts[1] and _MATH2 in ex.need_texts[1]
    assert _COOK not in ex.need_texts[1]


def test_build_examples_is_order_independent():
    """输入顺序打乱后，按时间排序的结果必须一致。"""
    rows = [(1, 10, 5, None), (1, 20, 1, None), (1, 30, 3, None)]
    a = build_examples(build_interactions(_bundle(rows)), {10: 0, 20: 1, 30: 2})
    b = build_examples(build_interactions(_bundle(list(reversed(rows)))),
                       {10: 0, 20: 1, 30: 2})
    assert a.need_texts == b.need_texts
    assert a.pos == b.pos


def test_build_examples_respects_max_need_items():
    """需求文本只保留最近 N 条交互，避免无限增长。"""
    rows = [(1, 10, t, None) for t in range(1, 8)] + [(1, 20, 99, None)]
    bundle = _bundle(rows)
    ex = build_examples(build_interactions(bundle), {10: 0, 20: 1}, max_need_items=2)
    assert len(ex) == 7
    assert _MATH2 not in ex.need_texts[-1]                 # 正样本自身不在需求里
    # 未触及上限的样本保留完整历史（第 1 个样本只有 1 条历史）
    assert ex.need_texts[0] == _MATH
    # 触及上限的样本恰好截到最近 2 条：第 7 个样本的历史是 7 条同文本
    assert ex.need_texts[-1] == f"{_MATH}{PREFIX_SEP}{_MATH}"
    # 对照：上限放到 10 时，同一个样本会保留完整历史
    ex_all = build_examples(build_interactions(bundle), {10: 0, 20: 1}, max_need_items=10)
    assert len(ex_all.need_texts[-1]) > len(ex.need_texts[-1])


def test_build_examples_skips_unknown_items():
    b = _bundle([(1, 10, 1, None), (1, 999, 2, None), (1, 20, 3, None)])
    ex = build_examples(build_interactions(b), {10: 0, 20: 1})
    assert all(p in (0, 1) for p in ex.pos)


def test_popular_negatives_picks_most_interacted():
    b = _bundle_with_repeat()
    ints = build_interactions(b)
    index = {10: 0, 20: 1, 30: 2, 40: 3}
    neg = popular_negatives(ints, index, top_n=2)
    assert len(neg) == 2
    assert set(neg).issubset({0, 1, 2, 3})


# --- 训练 ---

def _train(bundle=None, cfg=None):
    b = bundle or _bundle_with_repeat()
    cfg = cfg or _cfg()
    ints = build_interactions(b)
    item_ids = sorted(r.resource_id for r in b.resources)
    enc = LocalTfidfSvdEncoder(dim=cfg.encoder_dim, seed=cfg.seed)
    model = train_two_tower(enc, ints, item_ids, cfg)
    return model, enc, b, item_ids


def test_train_produces_learnable_model():
    model, _, _, _ = _train()
    assert len(model.loss_history) == 3
    assert model.loss_history[-1] < model.loss_history[0]      # loss 下降


def test_train_is_reproducible():
    m1, _, _, _ = _train()
    m2, _, _, _ = _train()
    assert np.allclose(m1.loss_history, m2.loss_history)
    p1 = torch.cat([p.flatten() for p in m1.parameters()])
    p2 = torch.cat([p.flatten() for p in m2.parameters()])
    assert torch.allclose(p1, p2)


def test_train_rejects_insufficient_data():
    b = _bundle([(1, 10, 1, None)])            # 每个用户只有 1 次交互
    cfg = _cfg()
    with pytest.raises(ValueError, match="至少需要 2 次交互"):
        train_two_tower(LocalTfidfSvdEncoder(dim=32), build_interactions(b),
                        [10, 20, 30, 40], cfg)


def test_training_prefers_same_cluster_items():
    """训练后，数学向需求的 top-1 应落在数学簇（语义 + 训练都指向它）。"""
    model, enc, b, item_ids = _train()
    item_texts = [next(r for r in b.resources if r.resource_id == i).description
                  for i in item_ids]
    prepare_item_matrix(model, enc, item_texts)

    math_user = "线性代数。矩阵与特征值。" + " " + _MATH + " " + _MATH2
    cook_user = "烹饪基础。家常菜与烘焙。" + " " + _COOK + " " + _COOK2
    scores = score_users(model, enc, [math_user, cook_user])
    assert scores.shape == (2, len(item_ids))

    math_top = item_ids[int(scores[0].argmax())]
    cook_top = item_ids[int(scores[1].argmax())]
    assert math_top in (10, 20)
    assert cook_top in (30, 40)


def test_hot_negative_items_change_training():
    m_plain, _, _, _ = _train(cfg=_cfg(recall_neg_items=0))
    m_neg, _, _, _ = _train(cfg=_cfg(recall_neg_items=2))
    p1 = torch.cat([p.flatten() for p in m_plain.parameters()])
    p2 = torch.cat([p.flatten() for p in m_neg.parameters()])
    assert not torch.allclose(p1, p2)


# --- 推理与融合 ---

def test_score_users_handles_empty_input():
    model, enc, b, item_ids = _train()
    prepare_item_matrix(model, enc, [r.description for r in b.resources])
    assert score_users(model, enc, []).shape == (0, len(item_ids))


def test_score_users_matches_rank():
    model, enc, b, item_ids = _train()
    item_texts = [next(r for r in b.resources if r.resource_id == i).description
                  for i in item_ids]
    prepare_item_matrix(model, enc, item_texts)
    query = _MATH
    scores = score_users(model, enc, [query])[0]
    _, idx = model.rank(torch.tensor(enc.encode([query]), dtype=torch.float32),
                        model.item_embs_cache)
    assert int(idx[0, 0]) == int(scores.argmax())


def test_build_examples_last_prefix_mode():
    """last_prefix：每个用户只用最后一个交互作正样本，样本数 = 用户数。"""
    rows = [(1, 10, 1, None), (1, 20, 2, None), (1, 30, 3, None), (2, 10, 4, None),
            (2, 30, 5, None)]
    bundle = _bundle(rows)
    item_index = {10: 0, 20: 1, 30: 2}
    ex = build_examples(build_interactions(bundle), item_index, mode="last_prefix")
    assert len(ex) == 2                                   # 两个用户各 1 个样本
    assert set(ex.user_of) == {1, 2}
    # 用户 1 的正样本是最后一条（30），需求由前两条拼成
    row = ex.user_of.index(1)
    assert ex.pos[row] == item_index[30]
    assert _MATH in ex.need_texts[row] and _MATH2 in ex.need_texts[row]
    assert _COOK not in ex.need_texts[row]                # 正样本自身不在需求里


def test_build_examples_rejects_unknown_mode():
    bundle = _bundle([(1, 10, 1, None), (1, 20, 2, None)])
    with pytest.raises(ValueError, match="样本构造模式"):
        build_examples(build_interactions(bundle), {10: 0, 20: 1}, mode="magic")


def test_build_examples_last_prefix_needs_two_interactions():
    bundle = _bundle([(1, 10, 1, None)])
    ex = build_examples(build_interactions(bundle), {10: 0}, mode="last_prefix")
    assert len(ex) == 0                                   # 只有 1 条交互 → 无历史可推


def test_fuse_scores_uses_config_weights():
    semantic = np.array([[0.0, 1.0]], dtype=np.float32)
    # 评分与热度都是常数 → 归一化后为 0（无区分度），只剩语义项按权重主导
    ratings = np.zeros(2, dtype=np.float32)
    popularity = np.zeros(2, dtype=np.float32)
    out = fuse_scores(semantic, ratings, popularity, _cfg())
    assert np.isclose(out[0, 0], 0.0)
    assert np.isclose(out[0, 1], _cfg().quality_w_semantic)   # 只剩语义项


def test_fuse_scores_quality_signals_break_ties():
    """语义持平时，评分/热度应当能把候选区分开。"""
    semantic = np.array([[0.5, 0.5]], dtype=np.float32)
    ratings = np.array([1.0, 5.0], dtype=np.float32)
    popularity = np.array([0.0, 10.0], dtype=np.float32)
    out = fuse_scores(semantic, ratings, popularity, _cfg())
    assert out[0, 1] > out[0, 0]
