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


def resource_text(resource: Resource) -> str:
    """资源文本 = 标题 + 正文（对齐 CourseHub 的 title + description）。

    文本构造属于「数据 → 模型」的入口口径，因此放在数据层，供 features / models /
    pipeline 共用，避免数据层反向依赖 features（原实现放在 encoder.py 里）。

    有些数据源（如 MOOCCube loader）已经把标题并进 `description`，此时不再重复拼接，
    否则会得到「线性代数。线性代数。矩阵…」这种重复文本，污染编码输入。
    """
    title = (resource.metadata or {}).get("title") or ""
    body = getattr(resource, "description", "") or ""
    if not body:
        return title
    if not title or body.startswith(title):
        return body
    return f"{title}。{body}"
