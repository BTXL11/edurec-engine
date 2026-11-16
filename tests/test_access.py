"""BundleAccess（索引化只读视图）的行为测试。

它必须与 encoder.user_need_text 的既有口径完全一致，否则「训练/评估/推理」
三处会产出不同的需求文本——这是最难排查的一类静默不一致。
"""

from engine.data.access import BundleAccess
from engine.data.schema import Behavior, DataBundle, Rating, Resource, User
from engine.features.encoder import user_need_text


def _bundle() -> DataBundle:
    resources = [
        Resource(resource_id=1, type="course", category_id=0,
                 metadata={"title": "线性代数"}, description="矩阵与特征值"),
        Resource(resource_id=2, type="course", category_id=0,
                 metadata={}, description="只有正文"),
        Resource(resource_id=3, type="course", category_id=1,
                 metadata={"title": "只有标题"}, description=""),
    ]
    users = [User(user_id=7), User(user_id=8)]
    behaviors = [
        Behavior(user_id=7, resource_id=1, action="view", ts=100),
        Behavior(user_id=7, resource_id=2, action="view", ts=200),
        Behavior(user_id=7, resource_id=1, action="view", ts=300),
        Behavior(user_id=7, resource_id=3, action="view", ts=400),
        Behavior(user_id=7, resource_id=99, action="view", ts=500),  # 未知资源
    ]
    ratings = [
        Rating(user_id=7, resource_id=1, score=5, ts=100),
        Rating(user_id=7, resource_id=1, score=3, ts=101),
        Rating(user_id=8, resource_id=2, score=4, ts=102),
    ]
    return DataBundle(users=users, resources=resources,
                      behaviors=behaviors, ratings=ratings)


def test_item_ids_are_sorted_and_texts_align():
    access = BundleAccess(_bundle())
    assert access.item_ids == [1, 2, 3]
    assert access.item_texts() == [
        "线性代数。矩阵与特征值",
        "只有正文",
        "只有标题",
    ]
    assert access.item_texts([3, 1]) == ["只有标题", "线性代数。矩阵与特征值"]


def test_history_is_chronological_and_seen_is_deduplicated():
    access = BundleAccess(_bundle())
    assert [b.ts for b in access.history(7)] == [100, 200, 300, 400, 500]
    assert access.seen(7) == {1, 2, 3, 99}
    assert access.history(8) == []
    assert access.seen(8) == set()


def test_popularity_and_mean_rating():
    access = BundleAccess(_bundle())
    assert access.popularity[1] == 2
    assert access.popularity[2] == 1
    assert access.mean_rating(1) == 4.0
    assert access.mean_rating(3) == 0.0


def test_need_text_matches_encoder_convention():
    bundle = _bundle()
    access = BundleAccess(bundle)
    # 最近 3 条不重复交互是 3、2、1；未知资源 99 被跳过
    assert access.need_text(7, max_items=3) == user_need_text(bundle, 7, max_items=3)
    assert access.need_text(8) == ""
    assert access.need_text(999) == ""


def test_need_texts_batch_matches_single():
    access = BundleAccess(_bundle())
    assert access.need_texts([7, 8], max_items=2) == [
        access.need_text(7, max_items=2), access.need_text(8, max_items=2)
    ]
