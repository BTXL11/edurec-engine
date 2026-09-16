from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class User:
    user_id: int


@dataclass(frozen=True)
class Resource:
    resource_id: int
    type: str                       # course | article | video
    category_id: int
    tags: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)
    description: str = ""           # 正文/简介，语义嵌入的主要文本来源
                                    # （平台非必需列，缺失时为空串）


@dataclass(frozen=True)
class Behavior:
    user_id: int
    resource_id: int
    action: str                     # view | click | favorite
    ts: int


@dataclass(frozen=True)
class Rating:
    user_id: int
    resource_id: int
    score: int                      # 1-5
    ts: int


@dataclass
class DataBundle:
    users: list[User]
    resources: list[Resource]
    behaviors: list[Behavior]
    ratings: list[Rating]
