"""MOOCCube 数据集加载器：把课程/概念/选课记录转成统一 DataBundle。

数据来源（本地旧版 MOOCCube，见 dataset/MOOCCube/MOOCCube/）：
  entities/course.json          课程（JSONL，一行一门课）：id/name/about/video_order…
  entities/user.json            用户（JSONL）：id/course_order/enroll_time（带时间戳）
  relations/course-concept.json 课程↔概念（TSV）
  entities/concept.json         概念：id/name/en/explanation

设计取舍：
  - **推荐单元是课程**，对齐 edurec-platform 的 resources 抽象与 CourseHub 的做法。
  - `about`（富文本）去标签后作为 `Resource.description`——语义嵌入的主要文本来源。
  - `category_id` 取自该课程概念的**学科领域**（概念 ID 后缀，如 `K_活性炭_化学` → `化学`），
    文本与类目因此天然对齐；无概念的课程归入「未分类」。
  - `tags` 取自课程概念名，用于词表/调试。
  - 数据集只有**选课行为、没有评分**，故 `ratings` 为空列表（诚实反映数据）。
  - 原始字符串 ID 无法直接进 vocab，用**排序后确定性编号**（id_map），
    原始 ID 留在 `metadata["source_id"]`，并可把映射落盘供交接时反查。
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from .schema import DataBundle, User, Resource, Behavior

_UNCATEGORIZED = "未分类"

# 概念 ID 形如 K_<概念名>_<学科领域>
_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")


def strip_html(text: str) -> str:
    """去掉富文本标签与 HTML 实体，压平空白；用于课程正文。"""
    if not text:
        return ""
    text = re.sub(r"<\s*(br|/p|/div|/li)\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\u00a0", " ")
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln).strip()


def parse_ts(value: str) -> int:
    """MOOCCube 的 `enroll_time` 形如 `2017-05-01 11:07:53`（UTC），转 Unix 秒。"""
    if not value:
        return 0
    value = value.strip()
    for fmt in _TS_FORMATS:
        try:
            dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    return 0


def build_id_map(source_ids) -> dict[str, int]:
    """字符串原始 ID → 连续整数编码：排序后编号，保证可复现。"""
    return {sid: i for i, sid in enumerate(sorted(set(source_ids)))}


def save_id_map(maps: dict[str, dict[str, int]], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(maps, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_id_map(path: str) -> dict[str, dict[str, int]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class MOOCCubeConfig:
    """MOOCCube 加载参数（与 EngineConfig 解耦，便于单测直接构造）。"""

    root: str = "dataset/MOOCCube/MOOCCube"
    max_users: int = 5000          # 活跃用户抽样上限（0 = 不限）
    min_user_courses: int = 5      # 视为「活跃」的最少选课数
    max_tags_per_course: int = 16  # 每门课保留的概念标签数上限
    include_concepts: bool = True  # 读 course-concept/concept 以生成类目与标签
    id_map_path: str = ""          # 非空则写出字符串 ID → 整数编码映射
    verbose: bool = False


def _log(cfg: "MOOCCubeConfig", msg: str) -> None:
    if cfg.verbose:
        print(f"[mooccube] {msg}", file=sys.stderr)


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_tsv(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0] and parts[1]:
                yield parts[0], parts[1]


def load_courses(root: str) -> dict[str, dict]:
    """课程 ID → {name, about, prerequisites}。"""
    courses: dict[str, dict] = {}
    path = os.path.join(root, "entities", "course.json")
    for obj in _iter_jsonl(path):
        cid = obj.get("id")
        if not cid:
            continue
        courses[cid] = {
            "name": (obj.get("name") or "").strip(),
            "about": strip_html(obj.get("about") or ""),
            "prerequisites": (obj.get("prerequisites") or "").strip(),
        }
    return courses


def apply_concepts(courses: dict[str, dict], root: str) -> None:
    """就地为课程补 category（学科领域）与 tags（概念名）。"""
    concept_field: dict[str, str] = {}
    concept_path = os.path.join(root, "entities", "concept.json")
    if os.path.isfile(concept_path):
        for obj in _iter_jsonl(concept_path):
            cid = obj.get("id")
            if cid:
                concept_field[cid] = cid.rsplit("_", 1)[-1] if "_" in cid else _UNCATEGORIZED

    course_concepts: dict[str, list[str]] = defaultdict(list)
    rel_path = os.path.join(root, "relations", "course-concept.json")
    if os.path.isfile(rel_path):
        for course_id, concept_id in _iter_tsv(rel_path):
            course_concepts[course_id].append(concept_id)

    for cid, course in courses.items():
        concept_ids = course_concepts.get(cid, [])
        fields = [concept_field[k] for k in concept_ids if k in concept_field]
        course["category"] = Counter(fields).most_common(1)[0][0] if fields else _UNCATEGORIZED
        names = []
        for k in concept_ids:
            body = k[2:] if k.startswith("K_") else k        # 去掉 K_ 前缀
            name = body.rsplit("_", 1)[0] if "_" in body else body
            if name:
                names.append(name)
        course["tags"] = tuple(dict.fromkeys(names))          # 去重且保序


def _select_users(user_path: str, user_course: dict[str, list[tuple[str, int]]],
                  cfg: MOOCCubeConfig) -> list[str]:
    """挑出「活跃用户」：选课数达阈值，最多 max_users 个；按 ID 排序保证可复现。

    抽样是**等间隔轮转**而非随机，避免引入与 seed 无关的抖动；
    `course_order` 的原始顺序保留（即用户的时间序），不做打乱。
    """
    candidates: list[str] = []
    for obj in _iter_jsonl(user_path):
        uid = obj.get("id")
        if not uid:
            continue
        order = obj.get("course_order") or []
        times = obj.get("enroll_time") or []
        if len(order) < cfg.min_user_courses:
            continue
        user_course[uid] = [
            (course_id, parse_ts(times[idx]) if idx < len(times) else 0)
            for idx, course_id in enumerate(order)
        ]
        candidates.append(uid)

    candidates.sort()
    _log(cfg, f"活跃用户（>= {cfg.min_user_courses} 门课）: {len(candidates)}")
    if cfg.max_users and len(candidates) > cfg.max_users:
        step = len(candidates) / float(cfg.max_users)
        candidates = [candidates[int(i * step)] for i in range(cfg.max_users)]
        _log(cfg, f"等间隔抽样至: {len(candidates)}")
    return candidates


def load(cfg: MOOCCubeConfig) -> DataBundle:
    """加载 MOOCCube → DataBundle（推荐单元为课程）。"""
    root = cfg.root
    course_path = os.path.join(root, "entities", "course.json")
    user_path = os.path.join(root, "entities", "user.json")
    if not os.path.isfile(course_path):
        raise FileNotFoundError(f"缺少课程文件: {course_path}（root 配置是否正确？）")
    if not os.path.isfile(user_path):
        raise FileNotFoundError(f"缺少用户文件: {user_path}")

    courses = load_courses(root)
    _log(cfg, f"课程: {len(courses)}")
    if cfg.include_concepts:
        apply_concepts(courses, root)
        _log(cfg, "课程类目（学科领域）: %d" % len({c["category"] for c in courses.values()}))
    else:
        for course in courses.values():
            course.setdefault("category", _UNCATEGORIZED)
            course.setdefault("tags", ())

    user_course: dict[str, list[tuple[str, int]]] = {}
    users = _select_users(user_path, user_course, cfg)

    course_ids = sorted(courses)
    user_ids = sorted(users)
    course_map = build_id_map(course_ids)
    user_map = build_id_map(user_ids)
    if cfg.id_map_path:
        save_id_map({"item2id": course_map, "user2id": user_map}, cfg.id_map_path)

    # 类目索引：按类目名排序编号，保证与文本对齐且可复现
    cat_names = sorted({c["category"] for c in courses.values()})
    cat2id = {name: i for i, name in enumerate(cat_names)}

    resources = []
    for cid in course_ids:
        c = courses[cid]
        title = c["name"] or cid
        description = f"{title}。{c['about']}" if c["about"] else title
        resources.append(Resource(
            resource_id=course_map[cid],
            type="course",
            category_id=cat2id[c["category"]],
            tags=tuple(c.get("tags", ()))[:cfg.max_tags_per_course],
            metadata={
                "source_id": cid,
                "title": title,
                "category": c["category"],
                "prerequisites": c["prerequisites"],
            },
            description=description,
        ))

    behaviors = []
    for uid in user_ids:
        for course_id, ts in user_course.get(uid, []):
            if course_id not in course_map:
                continue          # 选课记录指向课程表里不存在的课（数据集本身的空洞）
            behaviors.append(Behavior(
                user_id=user_map[uid],
                resource_id=course_map[course_id],
                action="view",
                ts=ts,
            ))
    _log(cfg, f"行为（选课）: {len(behaviors)}")

    users_out = [User(user_id=user_map[u]) for u in user_ids]
    return DataBundle(users=users_out, resources=resources,
                      behaviors=behaviors, ratings=[])
