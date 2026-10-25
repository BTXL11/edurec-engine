"""检查刚生成的 20 个 .ttl 的质量：每个课程的实体命中率与预计嵌入维度。"""
import glob
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import rdflib  # noqa: E402

from embeddings.embedder import load_model_, node_embeddings  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.join(ROOT, "output", "graphs", "v01")

model = load_model_(os.path.join(ROOT, "models", "wikipedia", "zhwiki_20180420_300d.txt"), 300)

files = sorted(glob.glob(os.path.join(GRAPH_DIR, "*.ttl")))
print(f"检查 {len(files)} 个图\n")
print(f"{'课程':46s} {'三元组':>7s} {'概念':>5s} {'命中':>5s} {'命中率':>7s} {'向量':>6s}")
print("-" * 88)

rows = []
for f in files:
    g = rdflib.Graph()
    g.parse(f, format="turtle")
    r = node_embeddings(model, g, method="wikipedia2vec")
    hits = len(r["concepts"])
    miss = len(r["missing_concepts"])
    total = hits + miss
    name = os.path.basename(f)[:-4]
    rate = (hits / total * 100) if total else 0.0

    # 复算这条课程的嵌入（与 embedder_embeddings 相同的加权平均）
    vec_dim = 0
    if hits:
        pr = np.sum([r["embeddings"][k]["pageRank"] for k in r["embeddings"]], 0)
        emb = np.sum([np.dot(r["embeddings"][k]["list"], r["embeddings"][k]["pageRank"])
                      for k in r["embeddings"]], 0)
        if pr:
            v = emb / pr
            vec_dim = np.asarray(v).shape[0]
            rows.append(v)

    short = name if len(name) <= 44 else name[:41] + "..."
    print(f"{short:46s} {len(g):7d} {total:5d} {hits:5d} {rate:6.1f}% {vec_dim:6d}")

print("-" * 88)
if rows:
    M = np.vstack(rows)
    print(f"\n可嵌入课程: {len(rows)}/{len(files)}")
    print(f"嵌入矩阵形状: {M.shape}  (期望 (N, 300))")
    print(f"含 NaN: {np.isnan(M).any()}   含 Inf: {np.isinf(M).any()}")
    print(f"向量范数: min={np.linalg.norm(M, axis=1).min():.2f} "
          f"max={np.linalg.norm(M, axis=1).max():.2f}")

    # 归一化后的余弦相似度，看看语义上是否合理
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    S = Mn @ Mn.T
    names = [os.path.basename(f)[:-4] for f in files][:len(rows)]
    print("\n最相似的 5 对课程（余弦）:")
    pairs = []
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            pairs.append((S[i, j], names[i], names[j]))
    for sim, a, b in sorted(pairs, reverse=True)[:5]:
        print(f"  {sim:.3f}  {a[:38]:40s} <-> {b[:38]}")
    print("\n最不相似的 3 对课程（余弦）:")
    for sim, a, b in sorted(pairs)[:3]:
        print(f"  {sim:.3f}  {a[:38]:40s} <-> {b[:38]}")
