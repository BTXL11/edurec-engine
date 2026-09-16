import json

import pytest

from engine.data.mooccube import (
    MOOCCubeConfig, build_id_map, load, load_id_map, parse_ts, strip_html,
)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_tsv(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for a, b in rows:
            f.write(f"{a}\t{b}\n")


def _make_dataset(root, users=None, courses=None, rels=None, field_of=None):
    (root / "entities").mkdir(parents=True, exist_ok=True)
    (root / "relations").mkdir(parents=True, exist_ok=True)

    courses = courses if courses is not None else [
        {"id": "C_a", "name": "线性代数", "about": "<p>矩阵与向量空间</p><p>特征值</p>",
         "prerequisites": "无", "video_order": [], "core_id": "C_a"},
        {"id": "C_b", "name": "Python 入门", "about": "<p>变量、循环与函数</p>",
         "prerequisites": "无", "video_order": [], "core_id": "C_b"},
        {"id": "C_c", "name": "无正文课程", "about": "", "prerequisites": "",
         "video_order": [], "core_id": "C_c"},
    ]
    users = users if users is not None else [
        {"id": "U_2", "course_order": ["C_a", "C_b", "C_a"], "enroll_time":
            ["2017-05-01 11:07:53", "2017-05-02 11:07:53", "2017-05-03 11:07:53"]},
        {"id": "U_1", "course_order": ["C_b", "C_c", "C_b", "C_b", "C_c"],
         "enroll_time": ["2018-01-01 00:00:00"] * 5},
        {"id": "U_3", "course_order": ["C_a"], "enroll_time": ["2018-01-01 00:00:00"]},
    ]
    rels = rels if rels is not None else [("C_a", "K_矩阵_数学"), ("C_a", "K_特征值_数学"),
                                          ("C_b", "K_函数_计算机科学")]
    field_of = field_of if field_of is not None else [
        ("K_矩阵_数学", "K_T_数学_数学"), ("K_特征值_数学", "K_T_数学_数学"),
        ("K_函数_计算机科学", "K_T_计算机科学_计算机科学")]

    _write_jsonl(root / "entities" / "course.json", courses)
    _write_jsonl(root / "entities" / "user.json", users)
    _write_jsonl(root / "entities" / "concept.json",
                 [{"id": cid, "name": cid, "en": "", "explanation": ""} for cid, _ in field_of])
    _write_tsv(root / "relations" / "course-concept.json", rels)


def test_strip_html():
    assert strip_html("<p>矩阵</p><p>特征值</p>") == "矩阵\n特征值"
    assert strip_html("<div>a &amp; b</div>") == "a & b"
    assert strip_html("") == ""
    assert strip_html("<p> 多个   空格 </p>") == "多个 空格"


def test_parse_ts():
    ts = parse_ts("2017-05-01 11:07:53")
    assert ts > 0
    assert parse_ts("2017-05-01 11:07:53") == parse_ts("2017-05-01 11:07:53")   # 确定性
    assert parse_ts("") == 0
    assert parse_ts("不是时间") == 0


def test_build_id_map_is_sorted_and_stable():
    m = build_id_map(["C_b", "C_a", "C_b"])
    assert m == {"C_a": 0, "C_b": 1}


def test_load_maps_to_bundle(tmp_path):
    _make_dataset(tmp_path)
    b = load(MOOCCubeConfig(root=str(tmp_path), min_user_courses=3, max_users=0))

    # 课程 → resource，原始 ID 保留在 metadata
    assert len(b.resources) == 3
    assert {r.metadata["source_id"] for r in b.resources} == {"C_a", "C_b", "C_c"}
    by_source = {r.metadata["source_id"]: r for r in b.resources}
    assert by_source["C_a"].type == "course"
    assert by_source["C_a"].description.startswith("线性代数。")
    assert "矩阵与向量空间" in by_source["C_a"].description
    assert "<p>" not in by_source["C_a"].description          # 标签被剥离
    assert by_source["C_c"].description == "无正文课程"        # 无正文时退回标题

    # 类目来自概念的学科领域，文本与类目对齐
    assert by_source["C_a"].metadata["category"] == "数学"
    assert by_source["C_b"].metadata["category"] == "计算机科学"
    assert by_source["C_c"].metadata["category"] == "未分类"
    assert by_source["C_a"].tags == ("矩阵", "特征值")

    # 只有活跃用户（>=3 门课）进入：U_1 与 U_2，U_3 被过滤
    assert len(b.users) == 2
    assert len(b.behaviors) == 8
    assert all(x.action == "view" for x in b.behaviors)
    assert all(x.ts > 0 for x in b.behaviors)

    # 数据集没有评分，诚实留空
    assert b.ratings == []


def test_max_users_interval_sampling(tmp_path):
    _make_dataset(tmp_path)
    b = load(MOOCCubeConfig(root=str(tmp_path), min_user_courses=3, max_users=1))
    assert len(b.users) == 1
    assert len({x.user_id for x in b.behaviors}) == 1


def test_load_is_reproducible(tmp_path):
    _make_dataset(tmp_path)
    cfg = MOOCCubeConfig(root=str(tmp_path), min_user_courses=3, max_users=0)
    a, b = load(cfg), load(cfg)
    assert [(r.resource_id, r.description) for r in a.resources] == \
           [(r.resource_id, r.description) for r in b.resources]
    assert [(x.user_id, x.resource_id, x.ts) for x in a.behaviors] == \
           [(x.user_id, x.resource_id, x.ts) for x in b.behaviors]


def test_id_map_written_and_reusable(tmp_path):
    _make_dataset(tmp_path)
    map_path = tmp_path / "model" / "id_map.json"
    load(MOOCCubeConfig(root=str(tmp_path), min_user_courses=3, max_users=0,
                        id_map_path=str(map_path)))
    assert map_path.is_file()
    maps = load_id_map(str(map_path))
    assert set(maps) == {"item2id", "user2id"}
    assert maps["item2id"]["C_a"] == 0
    assert maps["user2id"]["U_1"] == 0

    # 复用同一映射重新加载，编号不变
    b = load(MOOCCubeConfig(root=str(tmp_path), min_user_courses=3, max_users=0))
    by_source = {r.metadata["source_id"]: r.resource_id for r in b.resources}
    assert by_source["C_a"] == maps["item2id"]["C_a"]


def test_missing_course_file_raises(tmp_path):
    (tmp_path / "entities").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="缺少课程文件"):
        load(MOOCCubeConfig(root=str(tmp_path)))


def test_skips_behaviors_for_unknown_courses(tmp_path):
    _make_dataset(tmp_path, users=[
        {"id": "U_9", "course_order": ["C_a", "C_a", "C_不存在"],
         "enroll_time": ["2018-01-01 00:00:00"] * 3},
    ])
    b = load(MOOCCubeConfig(root=str(tmp_path), min_user_courses=3, max_users=0))
    assert len(b.behaviors) == 2
    assert all(x.resource_id != 999 for x in b.behaviors)
