"""通用资源流水线：输入课程/概念/描述 -> RDF 图谱 -> 向量嵌入。

与项目已有模块的关系（全部复用，不重写算法）：
    图谱    preprocessing/text.py  的 text2Wikifier + wikifier2RDF    （Wikifier 在线 API）
    嵌入    embeddings/tfidf.py, doc2vec.py, bert.py, embedder.py

设计要点
--------
1. **一个输入 = 一条资源**。字段都可选，只凭一个 `concept`（如"实变函数"）也能跑。
2. **构造用来 WiKify 的文本**：课程名 + 描述 + 概念词。纯概念输入也照样送，
   否则"实变函数"四个字只会抽出 4 个实体，向量信息量太低。
3. **嵌入方法可多选**，各自独立失败不影响其他方法（TF-IDF / Doc2Vec / BERT / EMBEDD-ER）。
4. **EMBEDD-ER 需要 .ttl 图**，所以它读的是刚生成的那张图；其余方法读文本。
5. 产出同时写一份 `resources.csv`（平台 `resources` 表格式），
   这样可以直接喂给 `Recommender.from_platform()` 做推荐。
"""

import json
import os
import re
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "model")

DEFAULT_METHODS = ["tfidf", "bert", "doc2vec"]


# --------------------------------------------------------------------------- #
# 输入定义
# --------------------------------------------------------------------------- #
class Resource:
    """一条待处理的资源（课程 / 概念 / 描述）。"""

    __slots__ = ("key", "title", "concepts", "description", "author", "tags",
                 "metadata", "source_url", "avg_rating", "view_count")

    def __init__(self, key=None, title="", concepts=None, description="",
                 author="", tags=None, metadata=None, source_url="",
                 avg_rating=0.0, view_count=0):
        self.key = key or _slug(title or (concepts[0] if concepts else "resource"))
        self.title = title or (concepts[0] if concepts else self.key)
        self.concepts = list(concepts or [])
        self.description = description or ""
        self.author = author or ""
        self.tags = list(tags or self.concepts)
        self.metadata = dict(metadata or {})
        self.source_url = source_url or ""
        self.avg_rating = float(avg_rating or 0.0)
        self.view_count = int(view_count or 0)

    def wikify_text(self):
        """拼出送去 Wikifier 的正文。

        概念词也拼进去：短输入（如"实变函数"）单靠标题抽不到几个实体。
        注意不要按"标题已在概念里就跳过"来做去重 —— 只给 `--concepts` 时
        title 会退化成 concepts[0]，那样其余概念就被丢了（实测 3 个概念只抽到 1 个）。
        """
        parts = [self.title]
        if self.description:
            parts.append(self.description)
        if self.concepts:
            parts.append("、".join(self.concepts))
        return "。".join(p for p in parts if p).strip()

    def embed_text(self):
        """拼出送去文本嵌入的内容（标题 + 描述 + **全部概念**）。

        注意不要把概念去重掉：只给了 `--concepts` 时 title 会退化成 concepts[0]，
        若按"title 已在 concepts 中就不重复"处理，其余概念就被丢了
        （实测 3 个概念只抽到 1 个的内容）。
        """
        parts = [self.title, self.description]
        if self.concepts:
            parts.append("、".join(self.concepts))
        return "。".join(p for p in parts if p).strip()

    def as_row(self, rid):
        """转成平台 resources 表的一行（`Recommender.from_platform` 可直接读）。"""
        md = dict(self.metadata)
        md.setdefault("concepts", self.concepts)
        return {
            "id": rid,
            "title": self.title,
            "description": self.description or "；".join(self.concepts),
            "type": "course",
            "category_id": 0,
            "tags": json.dumps(self.tags, ensure_ascii=False),
            "metadata": json.dumps(md, ensure_ascii=False),
            "author": self.author,
            "source_url": self.source_url,
            "avg_rating": round(self.avg_rating, 2),
            "view_count": self.view_count,
            "difficulty": "",
            "duration_minutes": 0,
        }


def _slug(text):
    """把标题转成可做文件名的 key。"""
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", str(text)).strip("_")
    return (s or "resource")[:60]


# --------------------------------------------------------------------------- #
# 各步骤
# --------------------------------------------------------------------------- #
def build_graph(resource, out_dir=None, threshold=-1, lang="zh", max_retries=3,
                verbose=True):
    """调用项目自己的 Wikifier 链路生成 RDF 图。

    返回 (rdflib.Graph | None, dict)。图会存成 <out_dir>/<key>.ttl（若给了 out_dir）。
    """
    from preprocessing.text import text2Wikifier, wikifier2RDF

    text = resource.wikify_text()
    t0 = time.time()
    try:
        data = text2Wikifier(text, threshold, lang=lang, max_retries=max_retries)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"    Wikifier 失败: {type(exc).__name__}: {exc}")
        return None, {"entities": 0, "seconds": round(time.time() - t0, 1),
                      "error": f"{type(exc).__name__}: {exc}"}

    g = wikifier2RDF(data)
    if g is None:
        if verbose:
            print(f"    Wikifier 未返回实体（{time.time()-t0:.1f}s）")
        return None, {"entities": 0, "seconds": round(time.time() - t0, 1)}

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, resource.key + ".ttl")
        g.serialize(format="turtle", destination=path)

    stats = {"entities": len(data or []), "triples": len(g),
             "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"    图谱: {stats['entities']} 实体 / {len(g)} 三元组 "
              f"({stats['seconds']}s)")
    return g, stats


def graph_concepts(g):
    """从图里取 (概念名 -> pageRank)。概念名用项目的 decode_concept 解码。"""
    from embeddings.embedder import decode_concept

    out = {}
    for s, p, o in g:
        if str(p) == "https://univ-nantes.fr/ontology/pageRank":
            out[decode_concept(s)] = float(o)
    return out


def embed_tfidf(texts):
    """TF-IDF 向量（中文友好）。

    不用 `embeddings/tfidf.py` 的默认分词：它用 sklearn 默认的 `token_pattern`
    （按空白切词），中文一句话会被当成一个 token —— 实测只有 1 条资源时
    维度退化成 1，向量无意义。这里显式用 **字符 n-gram**（中文在这种表示下最稳），
    同时关闭 `reduce`（原实现的 TSNE 降维会报错且是负优化）。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [str(t) for t in texts]
    # 单条资源时词表太小：把 n-gram 范围放大一点，保证有足够特征
    ngram = (2, 4) if len(docs) >= 2 else (1, 4)
    vec = TfidfVectorizer(analyzer="char", ngram_range=ngram,
                          sublinear_tf=True, min_df=1)
    return vec.fit_transform(docs).toarray().astype(np.float32)


def embed_doc2vec(texts):
    """Doc2Vec 向量。

    原实现建了 vocab 却不 train、且用英文分词会把中文删光，这里用项目里的 jieba
    分词重新训练（不引入新依赖，jieba 在 requirements 里）。
    """
    import jieba
    import gensim.models as g

    docs = [g.doc2vec.TaggedDocument([w for w in jieba.cut(t) if w.strip()], [i])
            for i, t in enumerate(texts)]
    epochs = 40 if len(docs) >= 10 else 80
    model = g.doc2vec.Doc2Vec(vector_size=300, min_count=1, epochs=epochs, seed=0)
    model.build_vocab(docs)
    model.train(docs, total_examples=len(docs), epochs=epochs)
    return np.vstack([model.dv[i] for i in range(len(docs))])


def embed_bert(texts, model_name="bert-base-chinese", use_cuda=None):
    """BERT 句向量，走项目 embeddings/bert.py 的 simpletransformers 链路。"""
    from embeddings.bert import bert_embeddings

    vecs = bert_embeddings(list(texts), model_name=model_name, use_cuda=use_cuda)
    return np.asarray(vecs, dtype=np.float32)


def embed_embedder(graph_rows):
    """EMBEDD-ER 向量：概念向量按 pageRank 加权平均（Wikipedia2Vec 中文模型）。

    graph_rows: [(概念名, pageRank), ...]，来自刚生成的图。
    """
    from embeddings.embedder import load_model_

    path = os.path.join(ROOT, "models", "wikipedia", "zhwiki_20180420_300d.txt")
    model = load_model_(path, 300)
    num = np.zeros(300)
    den = 0.0
    hit = miss = 0
    for concept, pr in graph_rows:
        try:
            v = model["ENTITY/" + concept]
        except KeyError:
            miss += 1
            continue
        if len(v) != 300:
            continue
        num += np.asarray(v, dtype=np.float64) * pr
        den += pr
        hit += 1
    if den == 0:
        raise ValueError(f"图上 {miss} 个概念全部查不到向量")
    return (num / den).astype(np.float32), {"hit": hit, "miss": miss}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(resources, methods=None, out_dir=None, threshold=-1, lang="zh",
        bert_model="bert-base-chinese", use_cuda=None, verbose=True):
    """对一组资源跑完整流水线，返回结果字典。"""
    methods = [m.lower() for m in (methods or DEFAULT_METHODS)]
    out_dir = out_dir or os.path.join(ROOT, "output", "pipeline")
    graph_dir = os.path.join(out_dir, "graphs")
    os.makedirs(out_dir, exist_ok=True)

    records = []
    t_all = time.time()

    # 1) 图谱（逐条，因为要调在线 API）
    if verbose:
        print(f"\n[1/3] 生成 RDF 图谱（{len(resources)} 条，Wikifier 在线调用）")
    for i, r in enumerate(resources, 1):
        if verbose:
            print(f"  [{i}/{len(resources)}] {r.key}  «{r.title}»")
        g, gstats = build_graph(r, graph_dir, threshold=threshold, lang=lang,
                                verbose=verbose)
        concepts = graph_concepts(g) if g is not None else {}
        records.append({"resource": r, "graph": g, "graph_stats": gstats,
                        "concepts": concepts})

    # 2) 文本嵌入（批量，一次编码整组更快）
    text = [r["resource"].embed_text() for r in records]
    vectors = {}
    detail = {}

    for m in methods:
        if verbose:
            print(f"\n[2/3] 嵌入: {m}")
        t0 = time.time()
        try:
            if m == "tfidf":
                mat = embed_tfidf(text)
                info = {"dim": int(mat.shape[1])}
            elif m == "doc2vec":
                mat = embed_doc2vec(text)
                info = {"dim": int(mat.shape[1])}
            elif m == "bert":
                mat = embed_bert(text, model_name=bert_model, use_cuda=use_cuda)
                info = {"dim": int(mat.shape[1]), "model": bert_model}
            elif m in ("embedder", "embeddings-em", "embedder-er"):
                # 逐条按各自图上的概念加权；缺图的记零向量
                rows = []
                hit = miss = 0
                for rec in records:
                    try:
                        v, st = embed_embedder(list(rec["concepts"].items()))
                        hit += st["hit"]
                        miss += st["miss"]
                    except ValueError:
                        v = np.zeros(300, dtype=np.float32)
                    rows.append(v)
                mat = np.vstack(rows)
                info = {"dim": int(mat.shape[1]), "concept_hit": hit,
                        "concept_miss": miss}
            else:
                if verbose:
                    print(f"    跳过未知方法 {m!r}（可选 tfidf/doc2vec/bert/embedder）")
                continue
            vectors[m] = mat
            info["seconds"] = round(time.time() - t0, 1)
            detail[m] = info
            if verbose:
                extra = {k: v for k, v in info.items() if k != "dim"}
                print(f"    OK 维度={info['dim']} "
                      + " ".join(f"{k}={v}" for k, v in extra.items()))
        except Exception as exc:  # noqa: BLE001
            detail[m] = {"error": f"{type(exc).__name__}: {exc}"}
            if verbose:
                print(f"    失败: {type(exc).__name__}: {str(exc)[:110]}")

    # 3) 落盘
    if verbose:
        print(f"\n[3/3] 写出结果 -> {os.path.relpath(out_dir, ROOT)}")

    summary = []
    for i, rec in enumerate(records, start=1):
        r = rec["resource"]
        item = {
            "key": r.key,
            "title": r.title,
            "concepts": r.concepts,
            "description": r.description,
            "author": r.author,
            "graph": {
                "entities": rec["graph_stats"].get("entities", 0),
                "triples": rec["graph_stats"].get("triples", 0),
                "ttl": (os.path.join("graphs", r.key + ".ttl")
                        if rec["graph"] is not None else None),
                "concepts": sorted(
                    ({"concept": c, "pageRank": round(pr, 6)}
                     for c, pr in rec["concepts"].items()),
                    key=lambda d: -d["pageRank"]),
            },
            "embeddings": {},
        }
        for m, mat in vectors.items():
            item["embeddings"][m] = {
                "dim": int(mat[i - 1].shape[0]),
                "vector": [round(float(x), 6) for x in mat[i - 1]],
            }
        summary.append(item)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methods": list(vectors),
        "method_detail": detail,
        "elapsed_seconds": round(time.time() - t_all, 1),
        "items": summary,
    }
    report_path = os.path.join(out_dir, "pipeline_result.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # resources.csv：平台表格式，可直接喂 Recommender.from_platform
    rows = [r["resource"].as_row(i) for i, r in enumerate(records, start=1)]
    if rows:
        import csv

        csv_path = os.path.join(out_dir, "resources.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # 概念表：便于人工核对每条资源抽到了什么
    concept_rows = []
    for i, rec in enumerate(records, start=1):
        for c, pr in rec["concepts"].items():
            concept_rows.append({"resource_id": i, "title": rec["resource"].title,
                                 "concept": c, "pageRank": round(pr, 6)})
    if concept_rows:
        import csv

        with open(os.path.join(out_dir, "concepts.csv"), "w", encoding="utf-8",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=["resource_id", "title", "concept",
                                              "pageRank"])
            w.writeheader()
            w.writerows(concept_rows)

    for rec in records:
        rec["graph"] = None      # 释放
    if verbose:
        print(f"    pipeline_result.json / resources.csv / concepts.csv / graphs/*.ttl")
        print(f"    共 {len(summary)} 条资源，耗时 {report['elapsed_seconds']}s")
    return report

