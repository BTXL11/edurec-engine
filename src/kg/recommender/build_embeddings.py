"""把所有已生成的 .ttl 图转成课程嵌入矩阵。

流程与 embeddings/embedder.py 的 embedder_embeddings 一致：
    图 -> pageRank 加权的概念向量平均 -> (300,) 课程向量

结果缓存在 rec_cache/embeddings.npz，避免每次排序都重新解析所有图。

用法:
    python -m recommender.build_embeddings          # 增量：只算没有缓存的
    python -m recommender.build_embeddings --force  # 全部重算
"""
import glob
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import rdflib  # noqa: E402

from embeddings.embedder import load_model_, node_embeddings  # noqa: E402
from recommender.idmap import (  # noqa: E402
    build_reverse_map,
    cid_from_filename,
    resolve,
)

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "model")
GRAPH_DIR = os.path.join(ROOT, "output", "graphs", "v01")
CACHE = os.path.join(ROOT, "rec_cache", "embeddings.npz")
MODEL_TXT = os.path.join(ROOT, "models", "wikipedia", "zhwiki_20180420_300d.txt")


def _load_model():
    return load_model_(MODEL_TXT, 300)


def save_embeddings(embeddings, path=CACHE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted(embeddings)
    if not keys:
        return None
    mat = np.vstack([np.asarray(embeddings[k], dtype=np.float32) for k in keys])
    np.savez_compressed(path, cids=np.array(keys, dtype=object), vectors=mat)
    return path


def load_embeddings(path=CACHE):
    """返回 ({course_id: np.ndarray}, [course_id])，无缓存时返回 ({}, [])。"""
    if not os.path.exists(path):
        return {}, []
    data = np.load(path, allow_pickle=True)
    cids = [str(c) for c in data["cids"]]
    vectors = data["vectors"]
    return {c: vectors[i] for i, c in enumerate(cids)}, cids


def build(force=False, verbose=True, limit=None):
    """扫描图目录，增量生成嵌入缓存。"""
    existing, _ = ({}, []) if force else load_embeddings()
    files = sorted(glob.glob(os.path.join(GRAPH_DIR, "*.ttl")))
    if limit:
        files = files[:limit]

    todo = [f for f in files if cid_from_filename(f) not in existing]
    if verbose:
        print(f"图文件 {len(files)} 个，已有缓存 {len(existing)} 个，待处理 {len(todo)} 个")

    if not todo:
        save_embeddings(existing)
        return existing

    model = _load_model()
    reverse = build_reverse_map()

    embeddings = dict(existing)
    t0 = time.time()
    failed = []
    for i, f in enumerate(todo, 1):
        cid = cid_from_filename(f)
        try:
            g = rdflib.Graph()
            g.parse(f, format="turtle")
            r = node_embeddings(model, g, method="wikipedia2vec")
            hits = len(r["concepts"])
            if hits == 0:
                # 没有任何概念命中：退化为零向量，并记录，避免静默产出垃圾
                failed.append((cid, "无概念命中"))
                embeddings[cid] = np.zeros(300, dtype=np.float32)
                continue
            pr = np.sum([r["embeddings"][k]["pageRank"] for k in r["embeddings"]], 0)
            if pr == 0:
                failed.append((cid, "pageRank 和为 0"))
                embeddings[cid] = np.zeros(300, dtype=np.float32)
                continue
            emb = np.sum([np.dot(r["embeddings"][k]["list"], r["embeddings"][k]["pageRank"])
                          for k in r["embeddings"]], 0)
            embeddings[cid] = np.asarray(emb / pr, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            failed.append((cid, f"{type(exc).__name__}: {exc}"))
            embeddings[cid] = np.zeros(300, dtype=np.float32)

        if verbose and (i % 20 == 0 or i == len(todo)):
            print(f"  [{i}/{len(todo)}] {time.time()-t0:.0f}s  {cid}  "
                  f"(命中 {hits} 个概念)")

    save_embeddings(embeddings)
    if verbose:
        print(f"完成: 共 {len(embeddings)} 门课程嵌入，用时 {time.time()-t0:.0f}s")
        print(f"缓存: {CACHE}")
        if failed:
            print(f"异常 {len(failed)} 条:")
            for cid, why in failed[:10]:
                print(f"  - {cid}: {why}")

    # 顺带统计覆盖率
    if verbose:
        pop = None
        try:
            from recommender.stats import load_cache

            pop = load_cache("popularity.json")
        except Exception:  # noqa: BLE001
            pass
        if pop:
            keys = {resolve(c, pop, reverse=reverse) for c in embeddings}
            hit = sum(1 for k in keys if k in pop)
            print(f"其中 {hit}/{len(embeddings)} 门有真实热度数据")
    return embeddings


if __name__ == "__main__":
    build(force="--force" in sys.argv)

