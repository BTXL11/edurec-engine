"""纯原项目方法的推荐效果评测。

只用 embeddings/ 下原项目的嵌入方法 + similarity/metrics.py 的余弦相似度，
**不引入** 热度 / 学生画像加权 / 先修约束 等后加信号。

指标：HitRate@5 / HitRate@10 / NDCG@10 / MRR@10（留一法，全量候选排序）。

数据轨道：MOOCCube
    候选集  = output/graphs/v01 已嵌入的课程（582 门）
    交互    = data/MOOCCube/relations/user-course.json（682,753 条）
    文本    = course.json 的 name + about + chapter + display_name
              （与 Generation.ipynb 构造 .ttl 图时的文本一致）

运行: .\.venv\Scripts\python.exe eval_original_methods.py
"""
import json
import os
import re
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from recommender.build_embeddings import load_embeddings  # noqa: E402
from recommender.idmap import build_reverse_map  # noqa: E402
from recommender.stats import load_courses, read_tsv_pairs  # noqa: E402
from similarity.metrics import cosine_similarity  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
N_USERS = 2000          # 评测用户数
MIN_INTER = 3           # 至少这么多门已嵌入课程才纳入
SEED = 0


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def strip_html(t):
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", str(t), flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&quot;", '"'), ("&#39;", "'")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def build_corpus(mode="full"):
    """返回 (向量顺序的 safe_cid 列表, 原始 id 列表, 文本列表)。

    mode 决定 Text 的拼法（原项目 Generation.ipynb 用的是 full）：
      full   name + about + chapter + display_name（原样，含章节编号与视频名）
      nocode full 去掉章节编号串（如 01.01.01.01）
      brief  只留 name + about（标题 + 简介正文）
    """
    emb, cids = load_embeddings()
    rev = build_reverse_map()

    def flatten(v):
        if isinstance(v, list):
            return " ".join(flatten(i) for i in v if i is not None)
        return str(v) if v is not None else ""

    texts_by_raw = {}
    for c in load_courses():
        if mode == "brief":
            parts = [flatten(c.get(k)) for k in ("name", "about")]
        else:
            parts = [flatten(c.get(k)) for k in ("name", "about", "chapter",
                                                 "display_name")]
        t = strip_html(". ".join(p for p in parts if p))
        if mode == "nocode":
            t = re.sub(r"(?:\b\d{2}(?:\.\d{2}){2,}\b\s*)+", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
        texts_by_raw[c["id"]] = t

    order = sorted(cids)
    raws = [rev.get(c) or c for c in order]
    texts = [texts_by_raw.get(r, "") for r in raws]
    return order, raws, texts


def build_interactions(raw_ids):
    """返回 {user: [原始 course id, ...]}，只保留在候选集里的交互。"""
    keep = set(raw_ids)
    out = {}
    for u, c in read_tsv_pairs("relations/user-course.json"):
        if c in keep:
            out.setdefault(u, []).append(c)
    return out


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def evaluate(vectors, raw_ids, interactions, n_users=N_USERS, seed=SEED):
    """留一法：留出每个用户 1 门，用其余已学课程均值向量排序全量候选。"""
    rng = np.random.default_rng(seed)
    idx_of = {c: i for i, c in enumerate(raw_ids)}
    M = np.vstack([np.asarray(v, dtype=np.float64) for v in vectors])
    # 预归一化，余弦 = 点积
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Mn = M / norms

    eligible = sorted(u for u, cs in interactions.items()
                      if len(set(cs)) >= MIN_INTER)
    if len(eligible) > n_users:
        eligible = list(rng.choice(eligible, size=n_users, replace=False))

    K = (5, 10)
    hits = {k: 0 for k in K}
    ndcg10 = 0.0
    mrr10 = 0.0
    n_eval = 0

    for u in eligible:
        seen = list(dict.fromkeys(interactions[u]))
        held = seen[-1]                       # 留最后一条
        rest = seen[:-1]
        prof_idx = [idx_of[c] for c in rest if c in idx_of]
        if not prof_idx or held not in idx_of:
            continue
        prof = Mn[prof_idx].mean(axis=0)
        pn = np.linalg.norm(prof)
        if pn == 0:
            continue
        scores = Mn @ (prof / pn)

        # 排除"已学过的训练部分"；留出项必须留在候选里，否则永远命中不了
        for c in rest:
            if c in idx_of:
                scores[idx_of[c]] = -np.inf

        top10 = np.argsort(-scores)[:10]
        gold = idx_of[held]
        pos = np.where(top10 == gold)[0]
        rank = int(pos[0]) + 1 if len(pos) else None

        n_eval += 1
        for k in K:
            if rank is not None and rank <= k:
                hits[k] += 1
        if rank is not None and rank <= 10:
            ndcg10 += 1.0 / np.log2(rank + 1)
            mrr10 += 1.0 / rank

    if n_eval == 0:
        return None
    return {
        "users": n_eval,
        "hr5": hits[5] / n_eval,
        "hr10": hits[10] / n_eval,
        "ndcg10": ndcg10 / n_eval,
        "mrr10": mrr10 / n_eval,
    }


# --------------------------------------------------------------------------- #
# 各嵌入方法
# --------------------------------------------------------------------------- #
def emb_tfidf_original(texts):
    """原 embeddings/tfidf.py 的字面行为。

    原实现 reduce=True 会调 TSNE(n_components=700)：582 篇文本时能跑通，
    但要求 perplexity(5) < n_samples，样本数不足时直接报错 —— 这里如实捕获。
    """
    from embeddings.tfidf import tfidf_embeddings
    return tfidf_embeddings(texts)


def emb_tfidf_noreduce(texts):
    """只把原实现的 reduce=True 关掉（即 tfidf_embeddings(texts, reduce=False)）。

    原函数的降维到 700 维对 582 篇文本既无必要也无意义，关掉它才拿得到真实 TF-IDF 向量。
    """
    from embeddings.tfidf import tfidf_embeddings
    return tfidf_embeddings(texts, reduce=False)


def emb_doc2vec_original(texts):
    """原 embeddings/doc2vec.py 的字面行为（含其预处理与缺点）。"""
    from embeddings.doc2vec import doc2vec_embeddings
    return doc2vec_embeddings(texts)


def emb_doc2vec_fixed(texts):
    """修正原实现的两个硬伤，其余保持原样。

    修正点（原实现：建了 vocab 但从不 train；预处理用英文分词+英文停用词
    且 `re.sub(r'[^\\w\\s]','')` 会把中文全删掉）：
      1. 用 jieba 分词而不是 word_tokenize；
      2. 真正调用 train()；
      3. 向量取训练后的文档向量，而不是未训练模型上的 infer_vector。
    """
    import jieba
    import gensim.models as g

    docs = [g.doc2vec.TaggedDocument([w for w in jieba.cut(t) if w.strip()], [i])
            for i, t in enumerate(texts)]
    model = g.doc2vec.Doc2Vec(vector_size=300, min_count=1, epochs=40, seed=0)
    model.build_vocab(docs)
    model.train(docs, total_examples=len(docs), epochs=40)
    return [model.dv[i] for i in range(len(docs))]


def emb_bert(texts, model_name, limit_chars=512):
    """原 embeddings/bert.py 的路径（simpletransformers RepresentationModel）。

    注意：原函数硬编码 model_name="bert-base-uncased"（英文词表，对中文几乎无效），
    所以这里通过原函数已参数化的 model_name 传入中文模型；预训练框架与调用链与原项目一致。
    """
    from embeddings.bert import bert_embeddings
    return bert_embeddings([t[:limit_chars] for t in texts], model_name=model_name)


def emb_embedder_er():
    """原项目的 EMBEDD-ER：Wikipedia2Vec 概念向量 + pageRank 加权聚合。

    这一列不是从文本现算的，而是本项目已生成好并缓存在 rec_cache/embeddings.npz 里的
    582 门课程向量（由 embeddings/embedder.py 的 embedder_embeddings 产出）。
    直接复用，保证与原项目产出完全一致。
    """
    emb, cids = load_embeddings()
    return [emb[c] for c in sorted(cids)]


# --------------------------------------------------------------------------- #
# 基线
# --------------------------------------------------------------------------- #
def evaluate_popularity(raw_ids, interactions, n_users=N_USERS, seed=SEED):
    """热门基线：按训练集出现频次排序（不含后加的 log1p/完成度等处理）。"""
    from collections import Counter

    rng = np.random.default_rng(seed)
    cnt = Counter()
    for u, cs in interactions.items():
        for c in set(cs):
            cnt[c] += 1
    idx_of = {c: i for i, c in enumerate(raw_ids)}
    rank_order = [c for c, _ in cnt.most_common() if c in idx_of]

    eligible = sorted(u for u, cs in interactions.items() if len(set(cs)) >= MIN_INTER)
    if len(eligible) > n_users:
        eligible = list(rng.choice(eligible, size=n_users, replace=False))

    hits = {5: 0, 10: 0}
    ndcg10 = mrr10 = 0.0
    n = 0
    for u in eligible:
        seen = list(dict.fromkeys(interactions[u]))
        held = seen[-1]
        rest = seen[:-1]
        if held not in idx_of:
            continue
        exclude = set(rest)          # 只排除训练部分，留出项留在候选里
        seq = [c for c in rank_order if c not in exclude]
        r = seq.index(held) + 1 if held in seq else None
        n += 1
        for k in (5, 10):
            if r is not None and r <= k:
                hits[k] += 1
        if r is not None and r <= 10:
            ndcg10 += 1.0 / np.log2(r + 1)
            mrr10 += 1.0 / r
    return {"users": n, "hr5": hits[5] / n, "hr10": hits[10] / n,
            "ndcg10": ndcg10 / n, "mrr10": mrr10 / n}


# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print("纯原项目方法的推荐效果评测（仅 embedding + 余弦相似度）")
    print("=" * 78)
    order, raws, texts = build_corpus("full")
    print(f"候选课程 {len(raws)} 门；文本长度 中位 {int(np.median([len(t) for t in texts]))} 字")
    inter = build_interactions(raws)
    elig = sum(1 for cs in inter.values() if len(set(cs)) >= MIN_INTER)
    print(f"交互用户 {len(inter)} 个，其中 >= {MIN_INTER} 门已嵌入课程的 {elig} 个")
    print(f"评测用 {N_USERS} 个用户（留一法），候选池 {len(raws)} 门")
    print()

    results = {}

    print("[1/8] 热门基线 ...")
    results["热门基线"] = evaluate_popularity(raws, inter)

    print("[2/8] 随机向量对照（下界参考）...")
    rng = np.random.default_rng(SEED)
    results["随机向量(下界)"] = evaluate(
        [rng.normal(size=64) for _ in texts], raws, inter)

    print("[3/8] EMBEDD-ER（原项目主方法，复用已产出的课程向量）...")
    try:
        results["EMBEDD-ER"] = evaluate(emb_embedder_er(), raws, inter)
        print("      OK")
    except Exception as e:  # noqa: BLE001
        results["EMBEDD-ER"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"      报错: {type(e).__name__}: {str(e)[:90]}")

    print("[4/8] 原 TF-IDF（字面调用，不关降维）...")
    try:
        v = emb_tfidf_original(texts)
        results["TF-IDF(原样)"] = evaluate(v, raws, inter)
        print("      OK")
    except Exception as e:  # noqa: BLE001
        results["TF-IDF(原样)"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"      报错: {type(e).__name__}: {str(e)[:90]}")

    print("[5/8] TF-IDF（仅关掉 reduce 降维）...")
    try:
        v = emb_tfidf_noreduce(texts)
        results["TF-IDF(不降维)"] = evaluate(v, raws, inter)
        print("      OK")
    except Exception as e:  # noqa: BLE001
        results["TF-IDF(不降维)"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"      报错: {type(e).__name__}: {str(e)[:90]}")

    print("[6/8] 原 Doc2Vec（字面调用）...")
    try:
        v = emb_doc2vec_original(texts)
        results["Doc2Vec(原样)"] = evaluate(v, raws, inter)
        print("      OK")
    except Exception as e:  # noqa: BLE001
        results["Doc2Vec(原样)"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"      报错: {type(e).__name__}: {str(e)[:90]}")

    print("[7/8] Doc2Vec（修正 vocab/train/中文分词）...")
    try:
        v = emb_doc2vec_fixed(texts)
        results["Doc2Vec(修正)"] = evaluate(v, raws, inter)
        print("      OK")
    except Exception as e:  # noqa: BLE001
        results["Doc2Vec(修正)"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"      报错: {type(e).__name__}: {str(e)[:90]}")

    print("[8/8] BERT（bert-base-chinese，走原 bert_embeddings 调用链）...")
    try:
        v = emb_bert(texts, "bert-base-chinese")
        results["BERT(中文)"] = evaluate(v, raws, inter)
        print("      OK")
    except Exception as e:  # noqa: BLE001
        results["BERT(中文)"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"      报错: {type(e).__name__}: {str(e)[:90]}")

    print()
    print("=" * 78)
    print(f"{'方法':20s} {'用户数':>7s} {'HR@5':>8s} {'HR@10':>8s} {'NDCG@10':>9s} {'MRR@10':>8s}")
    print("-" * 78)
    for name, r in results.items():
        if r is None:
            print(f"{name:20s} {'—':>7s} {'—':>8s} {'—':>8s} {'—':>9s} {'—':>8s}")
        elif "error" in r:
            print(f"{name:20s} 失败: {r['error'][:56]}")
        else:
            print(f"{name:20s} {r['users']:7d} {r['hr5']:8.4f} {r['hr10']:8.4f} "
                  f"{r['ndcg10']:9.4f} {r['mrr10']:8.4f}")
    print("=" * 78)

    # 文本构成归因（方法固定 TF-IDF 不降维）
    print()
    print("附加：Text 构成的影响（方法固定为 TF-IDF 不降维）")
    print(f"{'文本构成':24s} {'HR@5':>8s} {'HR@10':>8s} {'NDCG@10':>9s}")
    print("-" * 54)
    variants = {}
    for mode, label in (("full", "全文本(原样)"), ("nocode", "去章节编号"),
                        ("brief", "只留标题+简介")):
        _, _, ts = build_corpus(mode)
        r = evaluate(emb_tfidf_noreduce(ts), raws, inter)
        variants[label] = r
        print(f"{label:24s} {r['hr5']:8.4f} {r['hr10']:8.4f} {r['ndcg10']:9.4f}")

    out = os.path.join(ROOT, "rec_cache", "eval_original_methods.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": {"n_users": N_USERS, "min_inter": MIN_INTER,
                              "seed": SEED, "candidates": len(raws)},
                   "results": results, "text_variants": variants},
                  f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
