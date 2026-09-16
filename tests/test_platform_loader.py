import csv
import json
import pytest

from engine.data.platform import load, SNAPSHOT_VERSION


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _make_snapshot(root, contract_version=SNAPSHOT_VERSION):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"contract_version": contract_version, "run_id": "20260907_000000"}, f)
    _write_csv(root / "users.csv", ["user_id"], [[0], [3], [7]])
    _write_csv(root / "resources.csv",
               ["resource_id", "title", "type", "category_id", "tags_json",
                "metadata_json", "avg_rating", "view_count", "created_at"],
               [[101, "入门课", "course", 2, '["数学","入门"]', '{"level":"basic"}',
                 4.5, 12, 1750000000],
                [202, "视频", "video", 3, "[]", "{}", 0, 1, 1750000100]])
    _write_csv(root / "behaviors.csv",
               ["user_id", "resource_id", "action", "ts"],
               [[0, 101, "view", 1750000001], [0, 101, "click", 1750000002],
                [3, 202, "favorite", 1750000003]])
    _write_csv(root / "ratings.csv",
               ["user_id", "resource_id", "score", "ts"],
               [[3, 202, 5, 1750000004]])


def test_platform_loader_maps_columns(tmp_path):
    d = tmp_path / "snap1"
    _make_snapshot(d)
    b = load(str(d))

    assert [u.user_id for u in b.users] == [0, 3, 7]          # 原始 ID（含 0/空洞）
    assert [r.resource_id for r in b.resources] == [101, 202]

    r0 = b.resources[0]
    assert r0.type == "course" and r0.category_id == 2
    assert r0.tags == ("数学", "入门")                          # tags_json → tuple
    assert r0.metadata["title"] == "入门课"                     # 扩展列进 metadata
    assert r0.metadata["avg_rating"] == 4.5
    assert r0.metadata["created_at"] == 1750000000

    acts = [(x.user_id, x.resource_id, x.action, x.ts) for x in b.behaviors]
    assert (0, 101, "view", 1750000001) in acts
    assert [(x.user_id, x.resource_id, x.score) for x in b.ratings] == [(3, 202, 5)]


def test_platform_loader_rejects_contract_version(tmp_path):
    d = tmp_path / "snap2"
    _make_snapshot(d, contract_version=99)
    with pytest.raises(ValueError, match="契约版本"):
        load(str(d))


def test_platform_loader_missing_file(tmp_path):
    d = tmp_path / "snap3"
    _make_snapshot(d)
    (d / "ratings.csv").unlink()
    with pytest.raises(ValueError, match="缺少文件"):
        load(str(d))


def test_platform_loader_reads_description_when_exported(tmp_path):
    """platform 补导 description 列后，资源正文应进 Resource.description。"""
    d = tmp_path / "snap4"
    _make_snapshot(d)
    _write_csv(d / "resources.csv",
               ["resource_id", "title", "description", "type", "category_id",
                "tags_json", "metadata_json", "avg_rating", "view_count", "created_at"],
               [[101, "入门课", "从零讲解数学分析的核心概念", "course", 2,
                 '["数学","入门"]', "{}", 4.5, 12, 1750000000],
                [202, "视频", "", "video", 3, "[]", "{}", 0, 1, 1750000100]])
    b = load(str(d))
    assert b.resources[0].description == "从零讲解数学分析的核心概念"
    assert b.resources[1].description == ""


def test_platform_loader_without_description_column_degrades(tmp_path):
    """contract_version=1 的旧快照没有 description 列：不得报错，降级为空串。"""
    d = tmp_path / "snap5"
    _make_snapshot(d)
    b = load(str(d))
    assert all(r.description == "" for r in b.resources)


def test_platform_loader_handles_multiline_description(tmp_path):
    """正文里含换行/逗号/双引号时必须按标准 CSV 转义解析。

    engine 实测真实正文中 66% 含换行，因此这不是边界情况而是常态；
    若用「按行 split」或手工拼接解析，多行字段会被读坏。
    """
    d = tmp_path / "snap6"
    _make_snapshot(d)
    body = '第一段，含逗号\n第二段，含"引号"\n第三段'
    _write_csv(d / "resources.csv",
               ["resource_id", "title", "description", "type", "category_id",
                "tags_json", "metadata_json", "avg_rating", "view_count", "created_at"],
               [[101, "入门课", body, "course", 2, '["数学"]', "{}", 4.5, 12, 1750000000]])
    b = load(str(d))
    assert b.resources[0].description == body          # 换行与引号都被完整保留
    assert "\n" in b.resources[0].description


def test_platform_loader_row_count_survives_multiline_fields(tmp_path):
    """含多行正文时，行数仍应等于资源数（不能被按行读取读成更多行）。"""
    d = tmp_path / "snap7"
    _make_snapshot(d)
    rows = [[i, f"课程{i}", f"第一行\n第二行\n第三行", "course", 1, "[]", "{}",
             0.0, 0, 1750000000] for i in (101, 202, 303)]
    _write_csv(d / "resources.csv",
               ["resource_id", "title", "description", "type", "category_id",
                "tags_json", "metadata_json", "avg_rating", "view_count", "created_at"],
               rows)
    b = load(str(d))
    assert len(b.resources) == 3
    assert all(r.description.count("\n") == 2 for r in b.resources)
