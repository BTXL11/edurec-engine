"""消融实验：验证 α/β/γ 三项各自的真实贡献。

注意：留一法只在"已嵌入课程集合"内评估，而当前只有 20 门课嵌入，
候选项全部来自这 20 门，样本很小、抽样方差极大。这里的结果只能说明原型能跑通、
各项都参与了打分，**不能当作最终效果结论** —— 需要先跑完全部 583 门课。
更严格的对照（固定用户池）见 compare_scoring.py。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

from recommender.engine import DEFAULT_WEIGHTS, Recommender  # noqa: E402

print("加载 ...")
rec = Recommender(verbose=False)
print(f"可推荐课程 {len(rec.vec_by_raw)} 门\n")

configs = [
    ("仅相似度            (1,0,0)", {"sim": 1.0, "hot": 0.0, "prereq": 0.0}),
    ("仅热度              (0,1,0)", {"sim": 0.0, "hot": 1.0, "prereq": 0.0}),
    ("仅先修              (0,0,1)", {"sim": 0.0, "hot": 0.0, "prereq": 1.0}),
    ("相似+热度           (.78,.22,0)", {"sim": 0.78, "hot": 0.22, "prereq": 0.0}),
    ("相似+热度+先修(默认)", dict(DEFAULT_WEIGHTS)),
]

print(f"{'配置':26s} {'top-5 命中':>10s} {'top-5 MRR':>10s} {'top-10 命中':>11s} {'top-10 MRR':>10s}")
print("-" * 74)
results = {}
for label, w in configs:
    rec.weights.update(w)
    r5 = rec.evaluate_offline(n_users=300, top_k=5)
    r10 = rec.evaluate_offline(n_users=300, top_k=10)
    results[label] = (r5, r10)
    if r5["users_evaluated"] == 0:
        print(f"{label:26s} 无可用评估用户")
        continue
    print(f"{label:26s} {r5['hit_rate']:10.3f} {r5['mrr']:10.3f} "
          f"{r10['hit_rate']:11.3f} {r10['mrr']:10.3f}")

print("-" * 74)
base = results.get("相似+热度+先修(默认)")
if base and base[1]["users_evaluated"]:
    print(f"\n评估用户数: {base[1]['users_evaluated']}")
    print("说明: 候选项仅来自当前已嵌入的 20 门课程，样本很小；")
    print("      跑完全部 583 门课后再看这组数字才有意义。")
