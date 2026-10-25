"""严格对照：同一批用户、同一批候选，对比不同打分方式，并检查确定性。

目的：确认"仅相似度"的 hit_rate 变化到底是算法差异还是评估不稳定。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import numpy as np  # noqa: E402

from recommender.engine import NEUTRAL_PREREQ, Recommender  # noqa: E402

# 已被否决的门控版本参数，仅用于对照复现
PREREQ_GATE = 0.3
PREREQ_PENALTY = 0.8

rec = Recommender(verbose=False)
print(f"可推荐课程 {len(rec.vec_by_raw)} 门\n")


def make_pool(n_users=300, seed=0):
    """固定池子：确定性选出学过 >=2 门已嵌入课程的用户。"""
    rng = np.random.default_rng(seed)
    eligible = sorted(
        uid for uid, cs in rec.user_courses.items()
        if len([c for c in cs if c in rec.vec_by_raw]) >= 2
    )
    if len(eligible) > n_users:
        eligible = list(rng.choice(eligible, size=n_users, replace=False))
    return eligible


def eval_with(pool, scorer, top_k=5):
    hits = 0
    rr = 0.0
    for uid in pool:
        known = [c for c in rec.user_courses[uid] if c in rec.vec_by_raw]
        held = known[-1]
        rest = [c for c in known if c != held]
        uvec = np.mean(np.vstack([rec.vec_by_raw[c] for c in rest]), axis=0)
        lc = rec.learned_concepts(uid)
        cands = [c for c in rec.vec_by_raw if c not in set(rec.user_courses[uid])]
        if held not in cands:
            cands.append(held)
        scored = sorted(((scorer(uvec, lc, c), c) for c in cands), key=lambda t: -t[0])
        rank = next((i + 1 for i, (_, c) in enumerate(scored) if c == held), None)
        if rank and rank <= top_k:
            hits += 1
            rr += 1.0 / rank
    return hits / len(pool), rr / len(pool)


def sc_sim_only(uvec, lc, cid):
    return rec.similarity(uvec, cid) or 0.0


def sc_sim_plus_hot(uvec, lc, cid):
    return 0.75 * (rec.similarity(uvec, cid) or 0.0) + 0.22 * rec.popularity_of(cid)


def sc_sim_hot_gated(uvec, lc, cid):
    """当前默认逻辑：先修门控 + 微加权。"""
    base = 0.75 * (rec.similarity(uvec, cid) or 0.0) + 0.22 * rec.popularity_of(cid)
    from recommender.prerequisites import prerequisite_score

    p, _, _ = prerequisite_score(rec.graph, lc, cid)
    if p is None:
        return base
    if p < PREREQ_GATE:
        return base * PREREQ_PENALTY
    return base + 0.03 * p


def sc_sim_hot_linear(uvec, lc, cid):
    """旧默认逻辑：先修当线性加分项。"""
    from recommender.prerequisites import prerequisite_score

    p, _, _ = prerequisite_score(rec.graph, lc, cid)
    p = p if p is not None else NEUTRAL_PREREQ
    return 0.75 * (rec.similarity(uvec, cid) or 0.0) + 0.22 * rec.popularity_of(cid) + 0.03 * p


pool = make_pool(300, seed=0)
print(f"评估池: {len(pool)} 个用户（固定 seed=0）\n")

print(f"{'打分方式':30s} {'top-5 命中':>10s} {'top-5 MRR':>10s}")
print("-" * 54)
for label, fn in [
    ("仅相似度", sc_sim_only),
    ("相似度+热度", sc_sim_plus_hot),
    ("相似+热度+先修(线性)", sc_sim_hot_linear),
    ("相似+热度+先修(门控)", sc_sim_hot_gated),
]:
    hr, mrr = eval_with(pool, fn, top_k=5)
    print(f"{label:30s} {hr:10.3f} {mrr:10.3f}")

# 确定性检查：同一打分跑两遍
hr1, _ = eval_with(pool, sc_sim_only, 5)
hr2, _ = eval_with(pool, sc_sim_only, 5)
print(f"\n确定性检查（仅相似度跑两遍）: {hr1:.4f} vs {hr2:.4f} -> "
      f"{'一致' if hr1 == hr2 else '不一致！评估不稳定'}")

# 直接测 evaluate_offline 的稳定性
a = rec.evaluate_offline(n_users=300, top_k=5, seed=0)
rec.weights.update({"sim": 1.0, "hot": 0.0, "prereq": 0.0})
b = rec.evaluate_offline(n_users=300, top_k=5, seed=0)
print(f"\nevaluate_offline 默认权重: hit={a['hit_rate']:.4f} mrr={a['mrr']:.4f}")
print(f"evaluate_offline 仅相似度: hit={b['hit_rate']:.4f} mrr={b['mrr']:.4f}")
print("（若仅相似度低于默认，说明门控/惩罚在 sim-only 下仍在生效，属实现问题）")
