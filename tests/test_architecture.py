"""架构边界测试：离线推荐链路与在线 KG 服务不能互相渗透。

这是「engine 一个仓库、两个可部署单元」的机器可执行约束：

  * 离线链路（src/engine + train/infer 入口）不得 import HTTP/服务端模块，
    也不得 import src/kg —— 否则离线交付会隐式依赖在线能力；
  * KG 生产路径（server 之外）不得 import src/engine —— 两个能力域不共享代码；
  * 只有 scripts/kg_server.py 是网络入口。
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "engine"
OFFLINE_SCRIPTS = [
    ROOT / "scripts" / "train_semantic.py",
    ROOT / "scripts" / "run_batch_infer.py",
    ROOT / "scripts" / "gen_sim_data.py",
]
KG_PRODUCTION = [
    ROOT / "src" / "kg" / "generate.py",
    ROOT / "src" / "kg" / "recommender" / "pipeline.py",
    ROOT / "src" / "kg" / "preprocessing" / "text.py",
]

NETWORK_ROOTS = {"http", "urllib", "requests", "aiohttp", "socket", "fastapi", "flask"}


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _python_files(directory: Path):
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def test_offline_engine_has_no_network_or_kg_imports():
    for path in _python_files(ENGINE):
        roots = _import_roots(path)
        forbidden = (roots & NETWORK_ROOTS) | ({"kg"} if "kg" in roots else set())
        assert not forbidden, f"{path.relative_to(ROOT)} 不应依赖 {sorted(forbidden)}"


def test_offline_entrypoints_do_not_import_kg_server():
    for path in OFFLINE_SCRIPTS:
        source = path.read_text(encoding="utf-8")
        assert "kg.server" not in source and "from server" not in source, \
            f"{path.relative_to(ROOT)} 不得依赖 KG HTTP 服务"


def test_kg_production_path_does_not_import_engine():
    for path in KG_PRODUCTION:
        roots = _import_roots(path)
        assert "engine" not in roots, f"{path.relative_to(ROOT)} 不应 import engine"


def test_only_kg_server_wrapper_is_a_network_entrypoint():
    """scripts/ 下只有 kg_server.py 允许启动 HTTP 服务。"""
    for path in sorted((ROOT / "scripts").glob("*.py")):
        if path.name == "kg_server.py":
            continue
        source = path.read_text(encoding="utf-8")
        assert "http.server" not in source, f"{path.name} 不应启动 HTTP 服务"
