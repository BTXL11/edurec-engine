from __future__ import annotations
import csv
import json
import os
from .schema import DataBundle, User, Resource, Behavior, Rating

SNAPSHOT_VERSION = 1
_REQUIRED = ("users.csv", "resources.csv", "behaviors.csv", "ratings.csv")


def load(snapshot_dir: str) -> DataBundle:
    """加载 platform 导出的 v1 数据快照（目录内 meta.json + 5 张 csv）。

    ID 均为平台数据库原始 ID（含 0 起始等任意取值），不做重排。
    资源多余列（title/view_count/avg_rating/created_at…）保留到 metadata，
    供后续特征扩展使用。契约校验失败抛 ValueError。

    `description`（资源正文）是**可选列**：当前 contract_version=1 的快照不含它，
    缺失时降级为空串而不报错——语义模型所需正文由 platform 侧补导，
    详见 docs/platform-contract.md「方向 A · resources.csv」。
    """
    missing = [f for f in _REQUIRED if not os.path.isfile(os.path.join(snapshot_dir, f))]
    if missing:
        raise ValueError(f"快照缺少文件: {missing} @ {snapshot_dir}")

    meta_path = os.path.join(snapshot_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("contract_version") != SNAPSHOT_VERSION:
            raise ValueError(
                f"快照契约版本不匹配: {meta.get('contract_version')} != {SNAPSHOT_VERSION}")

    def _rows(name: str):
        with open(os.path.join(snapshot_dir, name), encoding="utf-8") as f:
            yield from csv.DictReader(f)

    users = [User(user_id=int(r["user_id"])) for r in _rows("users.csv")]
    resources = []
    for r in _rows("resources.csv"):
        tags_raw = (r.get("tags_json") or "").strip()
        tags = tuple(json.loads(tags_raw)) if tags_raw else ()
        metadata = {
            "title": r.get("title") or "",
            "view_count": int(float(r["view_count"])) if r.get("view_count") else 0,
            "avg_rating": float(r["avg_rating"]) if r.get("avg_rating") else 0.0,
            "created_at": int(r["created_at"]) if r.get("created_at") else 0,
        }
        try:
            extra = json.loads(r["metadata_json"]) if (r.get("metadata_json") or "").strip() else {}
            if isinstance(extra, dict):
                metadata.update(extra)
        except (json.JSONDecodeError, TypeError):
            pass
        resources.append(Resource(
            resource_id=int(r["resource_id"]),
            type=r["type"].strip(),
            category_id=int(r["category_id"]),
            tags=tags,
            metadata=metadata,
            description=(r.get("description") or "").strip(),
        ))
    behaviors = [Behavior(user_id=int(x["user_id"]),
                          resource_id=int(x["resource_id"]),
                          action=x["action"].strip(),
                          ts=int(x["ts"])) for x in _rows("behaviors.csv")]
    ratings = [Rating(user_id=int(x["user_id"]),
                      resource_id=int(x["resource_id"]),
                      score=int(x["score"]),
                      ts=int(x["ts"])) for x in _rows("ratings.csv")]
    return DataBundle(users=users, resources=resources,
                      behaviors=behaviors, ratings=ratings)
