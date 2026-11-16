"""生成 contracts/v1 的 golden fixtures（确定性，可重复运行）。

用法（engine 仓库根目录）：

    python contracts/v1/make_fixtures.py

产物会被 git 跟踪；改契约时重跑一次，再跑 `pytest tests/test_contract.py`。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
SNAPSHOT = os.path.join(FIXTURES, "snapshot")
INVALID = os.path.join(FIXTURES, "invalid")

RUN_ID = "20260101_000000"
EXPORTED_AT = 1767225600

USERS = [["1"], ["2"], ["3"]]

RESOURCES = [
    ["101", "线性代数", "向量、矩阵与特征值。", "course", "1", '["数学","线性代数"]',
     '{"level":"basic"}', "4.50", "120", "beginner", "600", "1767000000"],
    ["102", "Python 入门", "从变量到函数的基础教程。", "video", "2", '["编程","Python"]',
     '{}', "0.00", "80", "", "45", "1767000100"],
    ["103", "实变函数", "测度、可测函数与勒贝格积分。", "course", "1", '["数学","实变函数"]',
     '{"level":"advanced"}', "4.00", "30", "advanced", "900", "1767000200"],
]

BEHAVIORS = [
    ["1", "101", "view", "1767100000"],
    ["1", "101", "click", "1767100001"],
    ["1", "102", "favorite", "1767100002"],
    ["2", "102", "view", "1767100003"],
    ["2", "103", "view", "1767100004"],
    ["3", "101", "view", "1767100005"],
]

RATINGS = [
    ["1", "101", "5", "1767100010"],
    ["2", "103", "4", "1767100011"],
]

CATEGORIES = [
    ["1", "数学"],
    ["2", "计算机"],
]

RESOURCE_HEADER = [
    "resource_id", "title", "description", "type", "category_id", "tags_json",
    "metadata_json", "avg_rating", "view_count", "difficulty",
    "duration_minutes", "created_at",
]

RECOMMENDATIONS = {
    "1": [103, 102],
    "2": [101],
    "3": [102, 103],
}

ENVELOPE = {
    "contract_version": 1,
    "generated_at": 1767225700,
    "run_id": "20260101_000100",
    "snapshot_run_id": RUN_ID,
    "model": {
        "name": "semantic_deterministic_two_tower",
        "version": "v1",
        "encoder": "local-hashidf-svd",
    },
    "method": {"recall": "semantic", "rank": "quality_fusion"},
    "top_n": 20,
    "users_count": len(RECOMMENDATIONS),
    "quality_weights": {"semantic": 0.6, "rating": 0.0, "popularity": 0.4},
    "scores": {
        "1": [0.93, 0.62],
        "2": [0.88],
        "3": [0.71, 0.55],
    },
    "reasons": {
        "1": ["与你学过的《线性代数》相关", "根据你的学习记录推荐"],
        "2": ["与你学过的《实变函数》相关"],
        "3": ["热门课程", "与你的兴趣相关"],
    },
}


def _write_csv(path: str, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        # 显式用 LF：`csv.writer` 默认 CRLF，而 `.gitattributes` 会把 CSV 归一化为
        # LF；若按 CRLF 计算 meta.json 的 sha256，新克隆（LF）会校验失败。
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: str, payload, *, indent: int | None = None,
                bom: bool = False) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=indent)
    data = text.encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    with open(path, "wb") as f:
        f.write(data)


def build_snapshot() -> None:
    os.makedirs(SNAPSHOT, exist_ok=True)
    _write_csv(os.path.join(SNAPSHOT, "users.csv"), ["user_id"], USERS)
    _write_csv(os.path.join(SNAPSHOT, "resources.csv"), RESOURCE_HEADER, RESOURCES)
    _write_csv(os.path.join(SNAPSHOT, "behaviors.csv"),
               ["user_id", "resource_id", "action", "ts"], BEHAVIORS)
    _write_csv(os.path.join(SNAPSHOT, "ratings.csv"),
               ["user_id", "resource_id", "score", "ts"], RATINGS)
    _write_csv(os.path.join(SNAPSHOT, "categories.csv"),
               ["category_id", "name"], CATEGORIES)

    names = ["users.csv", "resources.csv", "categories.csv",
             "behaviors.csv", "ratings.csv"]
    meta = {
        "contract_version": 1,
        "run_id": RUN_ID,
        "exported_at": EXPORTED_AT,
        "behavior_window": {"start": 0, "end": 1767100005},
        "users_count": len(USERS),
        "resources_count": len(RESOURCES),
        "behaviors_count": len(BEHAVIORS),
        "ratings_count": len(RATINGS),
        "categories_count": len(CATEGORIES),
        "files_sha256": {n: _sha256(os.path.join(SNAPSHOT, n)) for n in names},
    }
    _write_json(os.path.join(SNAPSHOT, "meta.json"), meta, indent=2)


def build_recommendations() -> None:
    _write_json(os.path.join(FIXTURES, "recommendations.json"), RECOMMENDATIONS)
    _write_json(os.path.join(FIXTURES, "recommendations.meta.json"), ENVELOPE,
                indent=2)


def build_invalid() -> None:
    os.makedirs(INVALID, exist_ok=True)

    cases = []

    def add(name: str, target: str, payload, *, raw: bytes | None = None,
            indent: int | None = None, bom: bool = False) -> None:
        path = os.path.join(INVALID, name)
        if raw is not None:
            with open(path, "wb") as f:
                f.write(raw)
        else:
            _write_json(path, payload, indent=indent, bom=bom)
        cases.append({"file": name, "target": target})

    add("bom.json", "recommendations", {"1": [101]}, bom=True)
    add("string_id.json", "recommendations", {"1": ["101"]})
    add("float_id.json", "recommendations", {"1": [101.0]})
    add("negative_id.json", "recommendations", {"1": [-1]})
    add("extra_top_level.json", "recommendations",
        {"contract_version": 1, "1": [101]})
    add("non_numeric_key.json", "recommendations", {"user-1": [101]})
    add("not_object.json", "recommendations", [1, 2])
    add("duplicate_ids.json", "recommendations", {"1": [101, 101]})
    add("trailing_garbage.json", "recommendations", None,
        raw=b'{"1": [101]}{}')
    add("envelope_bad_scores.json", "envelope",
        {"contract_version": 1, "scores": {"1": ["high"]}})
    add("envelope_bad_version.json", "envelope",
        {"contract_version": "1"})

    _write_json(os.path.join(INVALID, "cases.json"), cases, indent=2)


def main() -> int:
    build_snapshot()
    build_recommendations()
    build_invalid()
    print(f"fixtures written under {FIXTURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
