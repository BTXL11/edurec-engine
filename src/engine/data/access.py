"""DataBundle 的索引化只读视图（read model）。

`DataBundle` 是扁平事实表：users / resources / behaviors / ratings 各是一张表。
训练、评估、推理都要反复按 user 找行为、按 resource 找文本与评分；如果每次都线性
扫描，复杂度会退化成 O(U×B) 与 O(R²)（全量推理、逐用户构造需求文本时尤其明显）。

本模块在入口把常用索引构建一次，之后所有查询都是 O(1) / O(该用户的交互数)。
它是**纯派生数据**，不拥有事实、不写回 bundle——因此可以安全地在任何读取场景复用。
"""

from __future__ import annotations

from collections import Counter

from .schema import Behavior, DataBundle, Resource, resource_text


class BundleAccess:
    """为 `DataBundle` 建立按用户/按资源的索引。

    典型用法：

        access = BundleAccess(bundle)
        item_texts = access.item_texts(item_ids)   # 顺序与 item_ids 一致
        need = access.need_text(user_id, max_items=20)
        seen = access.seen(user_id)

    注意：`behaviors_by_user` 内按 `(ts, resource_id)` 升序排列，去重/取最近时
    不要再假设输入顺序。
    """

    def __init__(self, bundle: DataBundle):
        self.bundle = bundle
        self.resources_by_id: dict[int, Resource] = {
            r.resource_id: r for r in bundle.resources
        }
        self._item_ids: list[int] = sorted(self.resources_by_id)
        self._text_by_id: dict[int, str] = {
            rid: resource_text(r) for rid, r in self.resources_by_id.items()
        }

        self.behaviors_by_user: dict[int, list[Behavior]] = {}
        for behavior in bundle.behaviors:
            self.behaviors_by_user.setdefault(behavior.user_id, []).append(behavior)
        for acts in self.behaviors_by_user.values():
            acts.sort(key=lambda x: (x.ts, x.resource_id))

        self.seen_by_user: dict[int, set[int]] = {
            user: {b.resource_id for b in acts}
            for user, acts in self.behaviors_by_user.items()
        }
        self.popularity: Counter[int] = Counter(
            b.resource_id for b in bundle.behaviors
        )

        self.ratings_by_item: dict[int, list[int]] = {}
        for rating in bundle.ratings:
            self.ratings_by_item.setdefault(rating.resource_id, []).append(rating.score)

    # --- 资源 ---

    @property
    def item_ids(self) -> list[int]:
        """全部资源 ID，升序（与编码矩阵行号一一对应）。"""
        return self._item_ids

    @property
    def text_by_id(self) -> dict[int, str]:
        return self._text_by_id

    def text(self, resource_id: int) -> str:
        """资源文本；未知 ID 返回空串（与旧实现一致，调用方自行过滤）。"""
        return self._text_by_id.get(resource_id, "")

    def item_texts(self, item_ids: list[int] | None = None) -> list[str]:
        """按给定 item_ids 顺序返回文本；缺省用升序全量 item_ids。"""
        ids = self._item_ids if item_ids is None else item_ids
        return [self._text_by_id.get(item, "") for item in ids]

    def mean_rating(self, resource_id: int) -> float:
        scores = self.ratings_by_item.get(resource_id)
        return sum(scores) / len(scores) if scores else 0.0

    # --- 用户行为 ---

    def history(self, user_id: int) -> list[Behavior]:
        """该用户的全部交互，按时间升序。"""
        return self.behaviors_by_user.get(user_id, [])

    def seen(self, user_id: int) -> set[int]:
        """该用户交互过的资源 ID 集合。"""
        return self.seen_by_user.get(user_id, set())

    def need_text(self, user_id: int, max_items: int = 20) -> str:
        """学生需求文本 = 最近 `max_items` 条**不重复**交互的资源文本拼接。

        与 `encoder.user_need_text` 的口径一致：按时间倒序取最近、同一资源只算一次、
        再按正序拼接。未知资源被跳过。
        """
        picked: list[str] = []
        seen: set[int] = set()
        for behavior in reversed(self.history(user_id)):
            rid = behavior.resource_id
            if rid in seen:
                continue
            text = self._text_by_id.get(rid)
            if text is None:
                continue
            seen.add(rid)
            picked.append(text)
            if len(picked) >= max_items:
                break
        return " ".join(reversed(picked))

    def need_texts(self, user_ids: list[int], max_items: int = 20) -> list[str]:
        """批量构造需求文本（顺序与 user_ids 一致）。"""
        return [self.need_text(user_id, max_items) for user_id in user_ids]
