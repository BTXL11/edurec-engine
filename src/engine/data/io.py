from __future__ import annotations
import os
import pandas as pd
from .schema import DataBundle, User, Resource, Behavior, Rating


def _df_to_resources(df: pd.DataFrame) -> list[Resource]:
    out = []
    for row in df.itertuples():
        desc = getattr(row, "description", "")      # 旧 CSV 无此列时降级为空串
        out.append(Resource(
            resource_id=int(row.resource_id),
            type=str(row.type),
            category_id=int(row.category_id),
            tags=tuple(str(row.tags).split("|")) if pd.notna(row.tags) else (),
            metadata={},
            description="" if desc is None or pd.isna(desc) else str(desc),
        ))
    return out


def save_bundle(bundle: DataBundle, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{"user_id": u.user_id} for u in bundle.users]).to_csv(
        os.path.join(out_dir, "users.csv"), index=False)
    pd.DataFrame([{
        "resource_id": r.resource_id, "type": r.type,
        "category_id": r.category_id, "tags": "|".join(r.tags),
        "description": r.description,
    } for r in bundle.resources]).to_csv(
        os.path.join(out_dir, "resources.csv"), index=False)
    pd.DataFrame([{
        "user_id": b.user_id, "resource_id": b.resource_id,
        "action": b.action, "ts": b.ts,
    } for b in bundle.behaviors]).to_csv(
        os.path.join(out_dir, "behaviors.csv"), index=False)
    pd.DataFrame([{
        "user_id": r.user_id, "resource_id": r.resource_id,
        "score": r.score, "ts": r.ts,
    } for r in bundle.ratings]).to_csv(
        os.path.join(out_dir, "ratings.csv"), index=False)


def load_bundle(in_dir: str) -> DataBundle:
    users = [User(user_id=int(r.user_id)) for r in pd.read_csv(os.path.join(in_dir, "users.csv")).itertuples()]
    res_df = pd.read_csv(os.path.join(in_dir, "resources.csv"))
    resources = _df_to_resources(res_df)
    behaviors = [Behavior(
        user_id=int(r.user_id), resource_id=int(r.resource_id),
        action=str(r.action), ts=int(r.ts),
    ) for r in pd.read_csv(os.path.join(in_dir, "behaviors.csv")).itertuples()]
    ratings = [Rating(
        user_id=int(r.user_id), resource_id=int(r.resource_id),
        score=int(r.score), ts=int(r.ts),
    ) for r in pd.read_csv(os.path.join(in_dir, "ratings.csv")).itertuples()]
    return DataBundle(users=users, resources=resources,
                      behaviors=behaviors, ratings=ratings)
