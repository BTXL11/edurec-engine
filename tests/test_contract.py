"""交接契约 v1 的端到端测试：fixtures、校验器、loader、写出器必须口径一致。"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.v1.validate import (  # noqa: E402
    validate_envelope,
    validate_recommendations,
    validate_snapshot,
)
from engine.data.platform import load  # noqa: E402
from engine.pipeline.infer_batch import (  # noqa: E402
    BatchResult,
    Recommendation,
    write_handoff,
)

FIXTURES = ROOT / "contracts" / "v1" / "fixtures"
SNAPSHOT = FIXTURES / "snapshot"
RECOMMENDATIONS = FIXTURES / "recommendations.json"
ENVELOPE = FIXTURES / "recommendations.meta.json"
INVALID = FIXTURES / "invalid"


def _invalid_cases():
    with open(INVALID / "cases.json", encoding="utf-8") as f:
        return json.load(f)


def test_snapshot_fixture_is_valid():
    assert validate_snapshot(str(SNAPSHOT)) == []


def test_recommendations_fixture_is_valid():
    assert validate_recommendations(str(RECOMMENDATIONS)) == []
    assert validate_envelope(str(ENVELOPE)) == []


@pytest.mark.parametrize("case", _invalid_cases(), ids=lambda c: c["file"])
def test_invalid_fixture_is_rejected(case):
    path = INVALID / case["file"]
    if case["target"] == "recommendations":
        errors = validate_recommendations(str(path))
    else:
        errors = validate_envelope(str(path))
    assert errors, f"{case['file']} 应该被拒绝，但校验通过了"


def test_kg_generate_fixture_matches_service_contract():
    with open(FIXTURES / "kg-generate.response.json", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["contract_version"] == 1
    assert isinstance(payload["title"], str) and payload["title"]
    assert payload["concepts"], "fixture 至少包含一个概念"
    for concept in payload["concepts"]:
        assert isinstance(concept["name"], str) and len(concept["name"]) >= 2
        assert isinstance(concept["page_rank"], (int, float))
        assert concept["page_rank"] >= 0


def test_envelope_alignment_is_checked():
    errors = validate_envelope(
        str(ENVELOPE),
        {"1": [103], "2": [101], "3": [102, 103]},  # 用户 1 的分数比主文件多一项
    )
    assert any("scores['1']" in e and "长度" in e for e in errors)


def test_platform_loader_accepts_contract_fixture():
    bundle = load(str(SNAPSHOT))
    assert [u.user_id for u in bundle.users] == [1, 2, 3]
    assert [r.resource_id for r in bundle.resources] == [101, 102, 103]
    assert bundle.resources[0].description == "向量、矩阵与特征值。"
    assert bundle.resources[1].description == "从变量到函数的基础教程。"
    assert [b.action for b in bundle.behaviors] == [
        "view", "click", "favorite", "view", "view", "view"]
    assert [(r.user_id, r.score) for r in bundle.ratings] == [(1, 5), (2, 4)]


def test_platform_loader_rejects_checksum_mismatch(tmp_path):
    copied = tmp_path / "snapshot"
    shutil.copytree(SNAPSHOT, copied)
    with open(copied / "users.csv", "a", encoding="utf-8") as f:
        f.write("4\n")
    with pytest.raises(ValueError, match="校验和不匹配"):
        load(str(copied))


def test_writer_output_passes_contract(tmp_path):
    result = BatchResult(recommendations={
        1: Recommendation(1, [103, 102], [0.9, 0.4]),
        2: Recommendation(2, [], []),
        3: Recommendation(3, [101], [0.7]),
    })
    main = tmp_path / "recommendations.json"
    envelope = tmp_path / "recommendations.meta.json"
    write_handoff(result, str(main), envelope_path=str(envelope), envelope={
        "contract_version": 1,
        "generated_at": 1767225700,
        "run_id": "20260101_000100",
        "snapshot_run_id": "20260101_000000",
        "model": {"name": "test", "version": "v1", "encoder": "local"},
        "top_n": 2,
        "users_count": 3,
    })

    assert validate_recommendations(str(main)) == []
    assert validate_envelope(str(envelope), json.loads(main.read_text("utf-8"))) == []
