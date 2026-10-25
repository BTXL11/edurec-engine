"""混合排序引擎：语义相似度 + 热度 + 学生画像 + 先修约束。

    Score(u, i) = α·Sim(u, i) + β·Hotness(i) + γ·PrereqMatch(u, i)

各项都在 [0,1]：
    Sim      用户画像向量与候选课程向量的余弦相似度（映射到 [0,1]）
    Hotness  真实观看行为算出的热度（recommender/popularity.py）
    Prereq   用户已学概念对候选课程先修概念的覆盖比例；无法判定时为中性值

冷启动：用户没有任何已嵌入课程时，Sim 记 0 并自动把权重让给热度，
避免"画像缺失 ⇒ 什么都推不出来"。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from recommender.build_embeddings import load_embeddings  # noqa: E402
from recommender.idmap import build_reverse_map, resolve  # noqa: E402
from recommender.prerequisites import (  # noqa: E402
    build_graph,
    course_prereq_concepts,
    prerequisite_score,
)
from recommender.profile import build_user_courses, build_user_vectors  # noqa: E402
from recommender.stats import load_cache  # noqa: E402

DEFAULT_WEIGHTS = {"sim": 0.75, "hot": 0.22, "prereq": 0.03}
# 无法判定先修时的中性分：不用 1.0（会白送分给无信息课程），也不用 0（会误伤新课程）
NEUTRAL_PREREQ = 0.5


def cos_to_unit(v):
    """余弦 [-1,1] -> [0,1]。"""
    return (float(v) + 1.0) / 2.0


# --------------------------------------------------------------------------- #
# 平台（B 站）侧数据装配
#
# 平台侧的交接物是「表导出」：resources 表全列 + user_behaviors 表全列，
# 资源 ID 是 B 站侧的整数主键（与 MOOCCube 的 C_course-v1:… 字符串 ID 无关）。
# 详见 DATA_FLOW.md。
# --------------------------------------------------------------------------- #
def tokenize_zh(text):
    """中文分词。用 jieba（requirements 里已有），过滤空白与单字标点。"""
    import jieba

    return [w for w in jieba.cut(str(text)) if w.strip() and not w.isdigit()]


def platform_popularity(resources):
    """把 resources 表的互动字段折算成 hotness ∈ [0,1]。

    与 recommender/popularity.py 的口径一致（log1p 压缩长尾 + 除自身最大值），
    只是数据源换成 resources 表的 view_count 与 metadata 里的 like/favorite。
    """
    import json
    import math

    metrics = {"view": [], "like": [], "favorite": []}
    parsed = {}
    for _, r in resources.iterrows():
        try:
            md = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else {}
        except (json.JSONDecodeError, TypeError):
            md = {}
        parsed[r["id"]] = md
        metrics["view"].append(float(r["view_count"] or 0))
        metrics["like"].append(float(md.get("like") or 0))
        metrics["favorite"].append(float(md.get("favorite") or 0))

    def lognorm(values):
        logged = [math.log1p(v) for v in values]
        m = max(logged) if logged else 0.0
        return [v / m if m > 0 else 0.0 for v in logged]

    nv = lognorm(metrics["view"])
    nl = lognorm(metrics["like"])
    nf = lognorm(metrics["favorite"])

    out = {}
    for i, rid in enumerate(resources["id"]):
        md = parsed[rid]
        out[rid] = {
            "hotness": round(0.5 * nv[i] + 0.2 * nl[i] + 0.3 * nf[i], 6),
            "view_count": int(metrics["view"][i]),
            "like": int(metrics["like"][i]),
            "favorite": int(metrics["favorite"][i]),
            "avg_rating": float(resources["avg_rating"].iloc[i] or 0),
            "typename": md.get("typename", ""),
        }
    return out


def platform_tfidf_embeddings(texts, max_features=None):
    """中文 TF-IDF 向量（字级 2-4 gram + jieba 词级），返回 list[np.ndarray]。

    不用 embeddings/tfidf.py 的 tfidf_embeddings()：它的 reduce=True 会 TSNE 到 700 维，
    37 个文档根本降不到 700 维，会直接报错；这里只取原始稀疏向量。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [str(t) for t in texts]
    vec = TfidfVectorizer(tokenizer=tokenize_zh, token_pattern=None,
                          max_features=max_features)
    mat = vec.fit_transform(texts).toarray()
    return [row.astype(np.float32) for row in mat]


class Recommender:
    def __init__(self, weights=None, verbose=True):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.verbose = verbose

        if verbose:
            print("加载数据 ...")
        self.popularity = load_cache("popularity.json") or {}
        self.embeddings, self.cids = load_embeddings()
        self.reverse = build_reverse_map()
        self.graph = build_graph(verbose=False)
        self.user_courses = build_user_courses(verbose=False)

        # 原始 id -> 课程向量（图文件名用的是 safe 名，需要统一到原始 id）
        self.vec_by_raw = {}
        self.raw_of_safe = {}
        for safe_cid, vec in self.embeddings.items():
            raw = resolve(safe_cid, self.popularity, reverse=self.reverse)
            key = raw or safe_cid
            self.raw_of_safe[safe_cid] = key
            self.vec_by_raw[key] = vec

        # 课程 -> 自身概念集合（用于把用户已学课程折算成概念集合）
        self.course_concepts = self.graph["course_concepts"]

        self.user_vectors = build_user_vectors(
            self.user_courses, self.vec_by_raw, verbose=False)

        if verbose:
            print(f"  可推荐课程 {len(self.vec_by_raw)} 门（有向量）")
            print(f"  有热度数据   {len(self.popularity)} 门")
            print(f"  用户画像     {len(self.user_vectors)} 个"
                  f"（共 {len(self.user_courses)} 个用户）")

    @classmethod
    def from_platform(cls, resources_csv, behaviors_csv, method="tfidf",
                      weights=None, verbose=True):
        """从平台侧的两个表导出装配推荐器（不读 MOOCCube、不读 rec_cache）。

        resources_csv : resources 表导出，需含 id/title/description/metadata/view_count
        behaviors_csv : user_behaviors 表导出，需含 user_id/resource_id/action
        method        : "tfidf" 或 "doc2vec"

        先修项（γ）在平台轨没有数据可算，全部记中性分，因此该轨道实际是
        Sim + Hotness 两项在起作用 —— 这是数据现状，不是实现取舍。
        """
        import pandas as pd

        res = pd.read_csv(resources_csv, encoding="utf-8")
        beh = pd.read_csv(behaviors_csv, encoding="utf-8")

        self = cls.__new__(cls)
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.verbose = verbose
        self.method = method

        # 文本 = 标题 + 正文（正文是语义召回的主要信号，见 DATA_FLOW.md §4.1）
        res["text"] = (res["title"].fillna("") + "。"
                       + res["description"].fillna("")).str.strip()

        if method == "tfidf":
            vecs = platform_tfidf_embeddings(res["text"].tolist())
        elif method == "doc2vec":
            import gensim.models as g

            tagdocs = [g.doc2vec.TaggedDocument(tokenize_zh(t), [i])
                       for i, t in enumerate(res["text"].tolist())]
            model = g.doc2vec.Doc2Vec(vector_size=100, min_count=1, epochs=60,
                                      seed=0)
            model.build_vocab(tagdocs)
            model.train(tagdocs, total_examples=len(tagdocs), epochs=60)
            vecs = [model.dv[i] for i in range(len(tagdocs))]
        else:
            raise ValueError(f"未知 method: {method!r}（可选 tfidf / doc2vec）")

        self.vec_by_raw = {int(rid): np.asarray(v, dtype=np.float32)
                           for rid, v in zip(res["id"], vecs)}
        self.resources = res
        self.raw_of_safe = {int(rid): int(rid) for rid in res["id"]}
        self.popularity = platform_popularity(res)

        # 行为 -> 用户历史（只把 view/click/favorite 一并当作正向交互）
        self.user_courses = {}
        for _, b in beh.iterrows():
            self.user_courses.setdefault(int(b["user_id"]), []).append(
                int(b["resource_id"]))
        self.behavior_rows = beh

        self.graph = {"course_concepts": {}}      # 平台轨无概念映射
        self.course_concepts = {}
        self._sim_mat = None
        self.user_vectors = build_user_vectors(
            self.user_courses, self.vec_by_raw, verbose=False)

        if verbose:
            n_desc = int((res["description"].fillna("").str.len() > 0).sum())
            print(f"  资源 {len(self.vec_by_raw)} 条"
                  f"（有正文 {n_desc} 条），嵌入方式 {method}"
                  f"，维度 {len(next(iter(self.vec_by_raw.values())))}")
            print(f"  行为 {len(beh)} 条，用户 {len(self.user_courses)} 个，"
                  f"其中有画像 {len(self.user_vectors)} 个")
            print(f"  热度：按 resources 的 view_count + metadata.like/favorite 计算")
            print(f"  先修：平台轨无数据，全记中性分 {NEUTRAL_PREREQ}")
        return self

    # ------------------------------------------------------------------ #
    # 各项分数
    # ------------------------------------------------------------------ #
    def popularity_of(self, raw_id):
        return self.popularity.get(raw_id, {}).get("hotness", 0.0)

    def set_resource_vectors(self, vectors_by_id):
        """替换候选向量并让相似度缓存失效。

        直接改 `vec_by_raw` 会让 `_sim_mat` 停用旧向量，评测结果会一直是旧嵌入的
        （同一次运行里换多种嵌入时踩过这个坑）。统一走这个方法。
        """
        self.vec_by_raw = {k: np.asarray(v, dtype=np.float32)
                           for k, v in vectors_by_id.items()}
        self._sim_mat = None
        self._sim_ids = None
        self._sim_idx = None
        self.user_vectors = build_user_vectors(
            self.user_courses, self.vec_by_raw, verbose=False)

    def _build_sim_cache(self):
        """把所有候选向量预先归一化成一个矩阵，供向量化计算余弦。

        原实现 `similarity()` 对每个候选单独做 numpy 运算，一次评测要调用上千万次，
        是主要的性能瓶颈（实测 800+ 秒 CPU 还没跑完）。这里预归一化后，
        一次矩阵乘法就能拿到所有候选的相似度。
        """
        cids = sorted(self.vec_by_raw)
        M = np.vstack([np.asarray(self.vec_by_raw[c], dtype=np.float64) for c in cids])
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._sim_ids = cids
        self._sim_idx = {c: i for i, c in enumerate(cids)}
        self._sim_mat = M / norms
        return self._sim_mat

    def similarity(self, user_vector, raw_id):
        vec = self.vec_by_raw.get(raw_id)
        if user_vector is None or vec is None:
            return None
        nu = np.linalg.norm(user_vector)
        nv = np.linalg.norm(vec)
        if nu == 0 or nv == 0:
            return None
        return cos_to_unit(float(np.dot(user_vector, vec) / (nu * nv)))

    def learned_concepts(self, user_id):
        """用户已学课程涉及的概念集合。

        带缓存：这个值只取决于用户的选课历史，跨不同权重配置完全相同。
        一次评测要跑十几组配置，不缓存就会把同一个集合运算重复十几遍
        （实测会把评测从秒级拖到分钟级直至卡死）。
        """
        cache = getattr(self, "_lc_cache", None)
        if cache is None:
            cache = self._lc_cache = {}
        hit = cache.get(user_id)
        if hit is None:
            concepts = set()
            for cid in self.user_courses.get(user_id, ()):
                concepts.update(self.course_concepts.get(cid, ()))
            hit = cache[user_id] = concepts
        return hit

    # ------------------------------------------------------------------ #
    # 主排序
    # ------------------------------------------------------------------ #
    def _score_one(self, user_vec, learned_concepts, cold, cid, w):
        """算单个候选项的分数。

        先修按线性加分项处理。曾试过"门控"版本（用户明显缺先修时乘惩罚系数），
        在 300 用户对照实验里 top-5 命中从 0.893 掉到 0.647，因此放弃。
        """
        sim = self.similarity(user_vec, cid)
        hot = self.popularity_of(cid)
        pscore, missing, need = prerequisite_score(self.graph, learned_concepts, cid)
        p = pscore if pscore is not None else NEUTRAL_PREREQ

        if cold:
            # 冷启动：把相似度权重让给热度，避免 Sim=0 把分数整体压掉
            base = (w["hot"] + w["sim"]) * hot
        else:
            base = w["sim"] * (sim if sim is not None else 0.0) + w["hot"] * hot

        return {
            "course_id": cid,
            "course_id_safe": self._safe_of(cid),
            "score": base + w["prereq"] * p,
            "sim": sim,
            "hotness": hot,
            "prereq": p,
            "prereq_known": pscore is not None,
            "prereq_missing": missing,
            "prereq_total": need,
            "cold_start": cold,
        }

    def recommend(self, user_id, top_k=10, exclude_learned=True,
                  weights=None, candidate_ids=None, explain=True):
        """给用户推荐课程。

        返回 [{course_id, score, sim, hotness, prereq, ...}, ...] 按分数降序。
        """
        w = dict(self.weights)
        if weights:
            w.update(weights)

        learned = set(self.user_courses.get(user_id, ()))
        user_vec = self.user_vectors.get(user_id)
        learned_concepts = self.learned_concepts(user_id) if user_id in self.user_courses else set()

        candidates = candidate_ids if candidate_ids is not None else list(self.vec_by_raw)
        if exclude_learned:
            candidates = [c for c in candidates if c not in learned]

        # 冷启动：没有画像就把相似度权重让给热度
        cold = user_vec is None

        rows = [self._score_one(user_vec, learned_concepts, cold, cid, w)
                for cid in candidates]

        rows.sort(key=lambda r: -r["score"])
        out = rows[:top_k] if top_k else rows
        if explain:
            self._print_table(user_id, out, cold)
        return out

    def _safe_of(self, raw_id):
        for safe, key in self.raw_of_safe.items():
            if key == raw_id:
                return safe
        return raw_id

    def _print_table(self, user_id, rows, cold):
        print(f"\n用户 {user_id}"
              + ("（冷启动：无已嵌入课程，权重已让给热度）" if cold else "")
              + f"  已学 {len(self.user_courses.get(user_id, []))} 门")
        print(f"{'课程':46s} {'总分':>6s} {'相似':>6s} {'热度':>6s} {'先修':>6s}  说明")
        print("-" * 108)
        for r in rows:
            sim = f"{r['sim']:.3f}" if r["sim"] is not None else "  -  "
            if r["prereq_known"]:
                pr = f"{r['prereq']:.2f}"
                note = f"先修 {r['prereq_total']-r['prereq_missing']}/{r['prereq_total']}"
            else:
                pr = f"{r['prereq']:.2f}"
                note = "无先修信息(中性)"
            print(f"{self._label(r['course_id']):46s} {r['score']:6.3f} {sim:>6s} "
                  f"{r['hotness']:6.3f} {pr:>6s}  {note}")

    def _label(self, course_id):
        """展示用名称。

        平台轨的 course_id 是整数主键，表是 titles；MOOCCube 轨是字符串课程 ID。
        两条轨道的 ID 体系不同，这里只影响展示，不参与排序。
        """
        if isinstance(course_id, (int, np.integer)) and hasattr(self, "resources"):
            try:
                t = self.resources.loc[self.resources["id"] == int(course_id), "title"].iloc[0]
                t = f"[{course_id}] {t}"
            except (IndexError, KeyError):
                t = f"[{course_id}]"
        else:
            t = str(course_id)
        return t if len(t) <= 44 else t[:43] + "…"

    def prereq_matrix(self, user_ids, pool, max_depth=5, verbose=False):
        """为给定用户集合预计算"先修满足度"向量，返回 {uid: np.ndarray}。

        先修项只取决于用户已学概念与候选课程，跨不同权重配置是同一个值。
        一次评测要跑多组权重，不预计算就会重复算好几遍。
        """
        from recommender.prerequisites import prerequisite_score

        out = {}
        for i, uid in enumerate(user_ids):
            lc = self.learned_concepts(uid)
            row = np.empty(len(pool))
            for j, cid in enumerate(pool):
                ps, _, _ = prerequisite_score(self.graph, lc, cid, max_depth=max_depth)
                row[j] = ps if ps is not None else NEUTRAL_PREREQ
            out[uid] = row
            if verbose and (i + 1) % 500 == 0:
                print(f"    先修矩阵 {i+1}/{len(user_ids)}")
        return out

    def evaluate_topk(self, n_users=2000, min_inter=3, seed=0, max_learned=None,
                      weights=None, candidates=None, prereq_rows=None):
        """标准的 top-K 排序指标：HR@5 / HR@10 / NDCG@10 / MRR@10。

        协议（与 eval_original_methods.py 完全一致，便于两种口径互比）：
          1. 取已学 >= min_inter 门候选课程的用户，随机抽 n_users 个（固定 seed）；
          2. 每个用户按其交互顺序留出**最后 1 门**作为测试目标；
          3. 用剩余已学课程的向量均值作为用户画像；
          4. 对候选池**全部课程**打分排序（留出项必须在候选池内，否则永远命中不了）；
          5. 只把"训练部分的已学课程"排除，留出项保留；
          6. 命中位置即 rank，据此算 HR/NDCG/MRR。

        注意：user-course.json 里没有时间戳，所以"最后 1 门"是文件顺序而非真实时序，
        这是个近似（详见文档说明）。
        """
        w = dict(self.weights)
        if weights:
            w.update(weights)

        from recommender.prerequisites import prerequisite_score

        rng = np.random.default_rng(seed)
        pool = candidates if candidates is not None else sorted(self.vec_by_raw)
        if getattr(self, "_sim_mat", None) is None:
            self._build_sim_cache()
        # 只保留有向量的候选。_sim_mat 是 (n_candidates, n_features)，已按行归一化。
        # 注意：不要在循环里做 `_sim_mat[cols]` —— 候选维度高时（如 TF-IDF 上万维）
        # 每个用户都会复制一份大矩阵，2000 用户能吃掉几十 GB 的分配量。
        # 改为按行归一化整表后直接按索引取行。
        pool = [c for c in pool if c in self._sim_idx]
        rows = np.array([self._sim_idx[c] for c in pool])
        hot = np.array([self.popularity_of(c) for c in pool])
        idx_of = {c: i for i, c in enumerate(pool)}
        row_of = {c: r for c, r in zip(pool, rows)}

        eligible = sorted(
            uid for uid, cs in self.user_courses.items()
            if len([c for c in set(cs) if c in idx_of]) >= min_inter)
        if len(eligible) > n_users:
            eligible = list(rng.choice(eligible, size=n_users, replace=False))

        hits = {5: 0, 10: 0}
        ndcg10 = mrr10 = 0.0
        n = 0
        for uid in eligible:
            seen = [c for c in dict.fromkeys(self.user_courses[uid]) if c in idx_of]
            if len(seen) < min_inter:
                continue
            held = seen[-1]
            rest = seen[:-1]
            if max_learned:
                rest = rest[:max_learned]
            if not rest:
                continue

            # 用户画像 = 已学课程向量的均值（在归一化空间里再归一化，等价于余弦方向）
            uvec = self._sim_mat[[row_of[c] for c in rest]].mean(axis=0)
            pn = np.linalg.norm(uvec)
            if pn == 0:
                continue
            # 一次矩阵乘法得到全部候选的余弦，再用 fancy indexing 取候选子集
            all_sim = self._sim_mat @ (uvec / pn)
            sim = (all_sim[rows] + 1.0) / 2.0          # 余弦 -> [0,1]

            # 先修项：优先用预计算矩阵，否则现算（缓存已生效）
            if prereq_rows is not None and uid in prereq_rows:
                pre = prereq_rows[uid]
            else:
                lc = self.learned_concepts(uid)
                pre = np.empty(len(pool))
                for i, cid in enumerate(pool):
                    ps, _, _ = prerequisite_score(self.graph, lc, cid)
                    pre[i] = ps if ps is not None else NEUTRAL_PREREQ

            scores = w["sim"] * sim + w["hot"] * hot + w["prereq"] * pre
            for c in rest:                              # 只排除训练部分
                scores[idx_of[c]] = -np.inf

            top10 = np.argsort(-scores)[:10]
            gold = idx_of[held]
            pos = np.where(top10 == gold)[0]
            rank = int(pos[0]) + 1 if len(pos) else None

            n += 1
            for k in (5, 10):
                if rank is not None and rank <= k:
                    hits[k] += 1
            if rank is not None:
                ndcg10 += 1.0 / np.log2(rank + 1)
                mrr10 += 1.0 / rank

        if n == 0:
            return {"users": 0, "note": "没有符合条件的用户"}
        return {
            "users": n,
            "hr5": hits[5] / n,
            "hr10": hits[10] / n,
            "ndcg10": ndcg10 / n,
            "mrr10": mrr10 / n,
            "candidates": len(pool),
            "weights": w,
        }

    def evaluate_offline(self, n_users=200, top_k=10, seed=0, max_learned=5):
        """离线评估（仅 HR@k + MRR，单个 top_k）。

        ⚠️ 与 `evaluate_topk` 的区别：这里**没有把训练部分的已学课程从候选里排除**，
        所以已学过的课程仍可能出现在推荐位并占用名额，指标偏乐观且口径不一致。
        要出正式指标请用 `evaluate_topk`。

        另外：候选集小时（例如只有 20 门课）抽样方差极大，所以这里先对 eligible
        排序再抽样，保证同 seed 结果可复现。这是原型评估，不是论文级指标。
        """
        rng = np.random.default_rng(seed)
        eligible = sorted(
            uid for uid, courses in self.user_courses.items()
            if len([c for c in courses if c in self.vec_by_raw]) >= 2
        )
        if len(eligible) > n_users:
            eligible = list(rng.choice(eligible, size=n_users, replace=False))
            eligible.sort()

        if not eligible:
            return {"users_evaluated": 0, "note": "没有用户同时学过 2 门以上已嵌入课程"}

        hits = 0
        rr_sum = 0.0
        for uid in eligible:
            known = [c for c in self.user_courses[uid] if c in self.vec_by_raw]
            held_out = known[-1]
            # 用去掉留出课程后的历史重建画像
            rest = [c for c in self.user_courses[uid] if c in self.vec_by_raw and c != held_out]
            rest = rest[:max_learned]
            uvec = np.mean(np.vstack([self.vec_by_raw[c] for c in rest]), axis=0)
            learned_concepts = self.learned_concepts(uid)
            cands = [c for c in self.vec_by_raw if c not in self.user_courses[uid]]
            if held_out not in cands:
                cands.append(held_out)

            scored = []
            for cid in cands:
                r = self._score_one(uvec, learned_concepts, False, cid, self.weights)
                scored.append((r["score"], cid))
            scored.sort(key=lambda t: -t[0])
            rank = next((i + 1 for i, (_, c) in enumerate(scored) if c == held_out), None)
            if rank is not None and rank <= top_k:
                hits += 1
                rr_sum += 1.0 / rank

        return {
            "users_evaluated": len(eligible),
            "top_k": top_k,
            "hit_rate": hits / len(eligible),
            "mrr": rr_sum / len(eligible),
            "note": "留一法，仅在已嵌入课程集合内评估",
        }


if __name__ == "__main__":
    rec = Recommender()
    print()
    print("=" * 104)
    print("① 有画像的用户（学过已嵌入课程）")
    print("=" * 104)

    # 找一个学过较多已嵌入课程的用户
    best = sorted(
        ((uid, len([c for c in cs if c in rec.vec_by_raw]))
         for uid, cs in rec.user_courses.items()),
        key=lambda t: -t[1])[:3]
    for uid, n in best:
        if n >= 1:
            rec.recommend(uid, top_k=8)

    print()
    print("=" * 104)
    print("② 冷启动用户演示（随机一个没学过已嵌入课程的用户）")
    print("=" * 104)
    for uid, cs in rec.user_courses.items():
        if not any(c in rec.vec_by_raw for c in cs):
            rec.recommend(uid, top_k=8)
            break

    print()
    print("=" * 104)
    print("③ 离线评估（留一法）")
    print("=" * 104)
    for k in (5, 10):
        res = rec.evaluate_offline(n_users=200, top_k=k)
        print(f"  top-{k}: {res}")
