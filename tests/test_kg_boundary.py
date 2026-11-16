"""边界测试：生产知识图谱路径可用，且不依赖 research/ 下的一次性脚本。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
KG = SRC / "kg"

PRODUCTION = [
    KG / "server.py",
    KG / "generate.py",
    KG / "recommender" / "pipeline.py",
    KG / "preprocessing" / "text.py",
]


def test_production_sources_do_not_reference_research():
    for path in PRODUCTION:
        text = path.read_text(encoding="utf-8")
        assert "research." not in text, f"{path} 不应依赖 research/"
        assert "from research" not in text, f"{path} 不应依赖 research/"


def test_server_module_imports_and_filters():
    sys.path.insert(0, str(SRC))
    try:
        server = importlib.import_module("kg.server")
        raw = [
            {"concept": "实变函数", "pageRank": 0.9},
            {"concept": "实变函数", "pageRank": 0.8},   # 去重
            {"concept": "的", "pageRank": 0.99},        # 停用词
            {"concept": "数学", "pageRank": 0.001},     # 低于阈值
            {"concept": "测度", "pageRank": 0.5},
        ]
        got = server.filter_concepts(raw, min_rank=0.005, max_concepts=30)
        assert [c["name"] for c in got] == ["实变函数", "测度"]
    finally:
        sys.path.remove(str(SRC))
