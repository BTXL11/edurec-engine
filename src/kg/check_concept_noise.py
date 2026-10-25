"""量化"泛概念噪声"：高文档频率的概念里有多少是无意义的。"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import numpy as np  # noqa: E402

from visualize_kg import load_course_graphs  # noqa: E402

cg = load_course_graphs(verbose=False)
n_doc = len(cg)
docs = {}
for cid, info in cg.items():
    for c in info["concepts"]:
        docs.setdefault(c, set()).add(cid)

df = Counter(len(v) for v in docs.values())
print(f"课程数 {n_doc}，概念总数 {len(docs)}\n")
print("按文档频率（被多少门课覆盖）分布:")
print(f"{'覆盖课程数':>10s} {'概念数':>8s} {'占比':>7s}")
for k in sorted(df, reverse=True):
    print(f"{k:>10d} {df[k]:>8d} {df[k]/len(docs)*100:>6.1f}%")

print("\n覆盖率 = 1.0（所有课程都命中）的概念，全部列出:")
allhit = sorted(c for c, d in docs.items() if len(d) == n_doc)
print(f"  共 {len(allhit)} 个:")
for c in allhit:
    print(f"    {c!r}")

print("\n覆盖率 >= 0.8 的概念数:", sum(1 for d in docs.values() if len(d) >= 0.8 * n_doc))

# 单字/标点类
import re

punct = [c for c in docs if len(c) <= 2 and not re.search(r"[\u4e00-\u9fff]{2}", c)]
one_char = [c for c in docs if len(c) == 1]
print(f"\n长度 1 的概念: {len(one_char)} 个，样例 {sorted(one_char)[:15]}")
print(f"长度 <=2 的: {sum(1 for c in docs if len(c) <= 2)} 个")

# 真正有区分度的概念（df 在 2~6 之间）
mid = [c for c, d in docs.items() if 2 <= len(d) <= 6]
print(f"\n有区分度的概念（2~6 门课覆盖）: {len(mid)} 个")
print("样例:", sorted(mid, key=lambda c: -len(docs[c]))[:20])
