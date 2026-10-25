"""混合排序引擎的推荐效果评测（与纯语义口径完全一致）。

评测协议与 eval_original_methods.py 相同：2000 用户、留最后 1 门、全量候选排序，
指标 HR@5 / HR@10 / NDCG@10 / MRR@10。

对比对象：
  A. 纯语义（相似度单项，即原项目做法）—— 直接复用 eval_original_methods.py 的结果
  B. 混合引擎  Score = α·Sim + β·Hotness + γ·Prereq（recommender/engine.py）
  C. 消融：逐项开关，看每个信号的真实贡献

运行: .\.venv\Scripts\python.exe eval_hybrid_engine.py
"""
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from recommender.engine import Recommender  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
N_USERS = 2000
MIN_INTER = 3
SEED = 0

W_SIM = {"sim": 1.0, "hot": 0.0, "prereq": 0.0}
W_HOT = {"sim": 0.0, "hot": 1.0, "prereq": 0.0}
W_PRE = {"sim": 0.0, "hot": 0.0, "prereq": 1.0}
W_DEFAULT = {"sim": 0.75, "hot": 0.22, "prereq": 0.03}


def row(r):
    if not r or r.get("users", 0) == 0:
        return "—"
    return (f"{r['users']:6d} {r['hr5']:8.4f} {r['hr10']:8.4f} "
            f"{r['ndcg10']:9.4f} {r['mrr10']:8.4f}")


def head(title):
    print()
    print("=" * 84)
    print(title)
    print("=" * 84)
    print(f"{'配置':26s} {'用户':>6s} {'HR@5':>8s} {'HR@10':>8s} {'NDCG@10':>9s} {'MRR@10':>8s}")
    print("-" * 84)


def main():
    print("加载混合引擎（MOOCCube 轨，EMBEDD-ER 向量 + 真实热度 + 概念先修）...")
    rec = Recommender(verbose=True)
    print()

    results = {}

    # --------------------------- 1. 主对比 --------------------------- #
    head("① 主对比：纯语义 vs 混合引擎（2000 用户，候选 582 门）")

    r_sim = rec.evaluate_topk(n_users=N_USERS, min_inter=MIN_INTER, seed=SEED,
                              weights=W_SIM)
    results["纯语义(仅Sim)"] = r_sim
    print(f"{'纯语义 (1,0,0)':26s} {row(r_sim)}")

    r_def = rec.evaluate_topk(n_users=N_USERS, min_inter=MIN_INTER, seed=SEED,
                              weights=W_DEFAULT)
    results["混合引擎(默认)"] = r_def
    print(f"{'混合引擎 (.75,.22,.03)':26s} {row(r_def)}")

    # --------------------------- 2. 单项 --------------------------- #
    head("② 单项贡献")
    for name, w in (("仅 Hotness (0,1,0)", W_HOT), ("仅 Prereq (0,0,1)", W_PRE)):
        r = rec.evaluate_topk(n_users=N_USERS, min_inter=MIN_INTER, seed=SEED,
                              weights=w)
        results[name] = r
        print(f"{name:26s} {row(r)}")

    # --------------------------- 3. 权重扫描 --------------------------- #
    head("③ 权重扫描（看 Sim/Hotness 取舍；γ 固定 0.03）")
    for a in (0.0, 0.2, 0.4, 0.6, 0.75, 0.9, 1.0):
        b = round((1.0 - a) * 0.88, 3)
        w = {"sim": a, "hot": b, "prereq": 0.03}
        r = rec.evaluate_topk(n_users=N_USERS, min_inter=MIN_INTER, seed=SEED,
                              weights=w)
        results[f"sim={a:.2f},hot={b:.2f}"] = r
        print(f"{'sim=%.2f hot=%.2f γ=0.03' % (a, b):26s} {row(r)}")

    # --------------------------- 4. 画像长度 --------------------------- #
    head("④ 画像长度的影响（默认权重）")
    for ml in (3, 5, 10, None):
        r = rec.evaluate_topk(n_users=N_USERS, min_inter=MIN_INTER, seed=SEED,
                              weights=W_DEFAULT, max_learned=ml)
        label = "全部已学" if ml is None else f"取前 {ml} 门"
        results[f"画像={label}"] = r
        print(f"{label:26s} {row(r)}")

    # --------------------------- 5. 与纯语义其他嵌入对比 --------------------------- #
    head("⑤ 混合引擎换嵌入（默认权重）—— 对照纯语义同名方法")
    try:
        import eval_original_methods as EO

        _, raws, texts = EO.build_corpus("full")
        _, _, brief = EO.build_corpus("brief")
        variants = {
            "Doc2Vec(修正)": lambda: EO.emb_doc2vec_fixed(texts),
            "TF-IDF(仅标题+简介)": lambda: EO.emb_tfidf_noreduce(brief),
            "BERT(中文)": lambda: EO.emb_bert(texts, "bert-base-chinese"),
        }
        for name, fn in variants.items():
            try:
                vecs = fn()
                rec.set_resource_vectors(dict(zip(raws, vecs)))
                r = rec.evaluate_topk(n_users=N_USERS, min_inter=MIN_INTER,
                                      seed=SEED, weights=W_DEFAULT)
                results[f"混合+{name}"] = r
                print(f"{name:26s} {row(r)}")
            except Exception as e:  # noqa: BLE001
                print(f"{name:26s} 失败: {type(e).__name__}: {str(e)[:40]}")
    except Exception as e:  # noqa: BLE001
        print(f"  跳过（{type(e).__name__}: {e}）")

    # --------------------------- 6. 参考基线 --------------------------- #
    head("⑥ 参考基线（同协议，来自 eval_original_methods.py）")
    try:
        _, raws, _ = EO.build_corpus("full")
        inter = EO.build_interactions(raws)
        rnd = EO.evaluate(np.random.default_rng(SEED).normal(size=(len(raws), 64)),
                          raws, inter, n_users=N_USERS)
        results["随机向量(下界)"] = rnd
        print(f"{'随机向量(下界)':26s} {row(rnd)}")
        pop = EO.evaluate_popularity(raws, inter, n_users=N_USERS)
        results["热门基线"] = pop
        print(f"{'热门基线':26s} {row(pop)}")
    except Exception as e:  # noqa: BLE001
        print(f"  跳过（{type(e).__name__}: {e}）")

    out = os.path.join(ROOT, "rec_cache", "eval_hybrid_engine.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": {"n_users": N_USERS, "min_inter": MIN_INTER,
                              "seed": SEED,
                              "candidates": len(rec.vec_by_raw)},
                   "results": results}, f, ensure_ascii=False, indent=2)
    print()
    print(f"结果已写入 {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
