"""数据加载与统计工具（供推荐原型使用）。

MOOCCube 的这些 .json 其实大多是 JSON Lines；relations/ 下的是 2 列 TSV 而不是 JSON，
容易踩坑，所以统一在这里处理，避免各处重复出错。
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "model")
DATA = os.path.join(ROOT, "data", "MOOCCube")
# 注意：data/ 目录在本机沙箱下不可写（install 依赖目录之一），
# 所以处理结果缓存在项目根下的 rec_cache/。
PROCESSED = os.path.join(ROOT, "rec_cache")


# --------------------------------------------------------------------------- #
# 基础文件工具
# --------------------------------------------------------------------------- #
def file_size(path):
    return os.path.getsize(path)


def count_lines(path, block=1 << 22):
    """快速统计行数并返回耗时，用于评估读取策略。"""
    import time

    t0 = time.time()
    n = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            n += chunk.count(b"\n")
    return n, time.time() - t0


# --------------------------------------------------------------------------- #
# 各种格式的读取器
# --------------------------------------------------------------------------- #
def read_tsv_pairs(rel):
    """读 relations/ 下的 2 列 TSV，返回 [(col0, col1), ...]。"""
    out = []
    path = os.path.join(DATA, rel)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                out.append((parts[0].strip(), parts[1].strip()))
    return out


def iter_jsonl(rel):
    """按行流式读 JSON Lines，逐条 yield dict（不整体载入内存）。"""
    path = os.path.join(DATA, rel)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_courses():
    """course.json: id/name/prerequisites/about/core_id/video_order/display_name/chapter"""
    return list(iter_jsonl("entities/course.json"))


# --------------------------------------------------------------------------- #
# 缓存读写（大文件聚合结果缓存成小文件，避免重复读 1.7 GB）
# --------------------------------------------------------------------------- #
def save_cache(name, obj):
    os.makedirs(PROCESSED, exist_ok=True)
    path = os.path.join(PROCESSED, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    return path


def load_cache(name, default=None):
    path = os.path.join(PROCESSED, name)
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

