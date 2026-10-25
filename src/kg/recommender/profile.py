"""用户画像：从选课历史构造用户向量。

数据源：relations/user-course.json（682,753 条选课，48,640 个用户）

画像方式：把用户已学课程的 EMBEDD-ER 向量做均值池化（原计划就是先均值、后升级 GRU）。
只有已经生成过 .ttl 图的课程才有向量，所以覆盖率取决于已跑课程数 —— 冷启动用户
（一门已跑课程都没学过）返回 None，由排序层退回纯热度推荐。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from recommender.stats import PROCESSED, load_cache, read_tsv_pairs, save_cache  # noqa: E402

CACHE_NAME = "user_courses.json"


def build_user_courses(verbose=True):
    """user_id -> [course_id, ...]（原始 id）。"""
    cached = load_cache(CACHE_NAME)
    if cached:
        if verbose:
            print(f"  使用缓存画像数据（{len(cached)} 个用户）")
        return cached

    pairs = read_tsv_pairs("relations/user-course.json")
    user_courses = {}
    for uid, cid in pairs:
        user_courses.setdefault(uid, []).append(cid)

    # 课程 -> 选课用户数，方便看数据分布
    save_cache(CACHE_NAME, user_courses)
    if verbose:
        print(f"  选课记录 {len(pairs)} 条，用户 {len(user_courses)} 个")
    return user_courses


def build_user_vectors(user_courses, embeddings, min_courses=1, verbose=True):
    """均值池化出用户向量。

    user_courses: {user_id: [course_id, ...]}（原始 id）
    embeddings:   {course_id: np.ndarray}（原始 id -> 向量）
    返回 {user_id: np.ndarray}，只包含至少覆盖 min_courses 门已嵌入课程的用户。
    """
    user_vectors = {}
    for uid, courses in user_courses.items():
        vecs = [embeddings[c] for c in courses if c in embeddings]
        if len(vecs) >= min_courses:
            user_vectors[uid] = np.mean(np.vstack(vecs), axis=0)
    if verbose:
        cov = len(user_vectors) / max(len(user_courses), 1) * 100
        print(f"  可构造画像的用户: {len(user_vectors)}/{len(user_courses)} ({cov:.1f}%)")
    return user_vectors


def coverage_report(user_courses, embeddings):
    """画像覆盖率报告：已嵌入课程能覆盖多少选课记录。"""
    total = 0
    covered = 0
    for courses in user_courses.values():
        for c in courses:
            total += 1
            if c in embeddings:
                covered += 1
    return {"total_enrollments": total, "covered_enrollments": covered,
            "coverage": covered / total if total else 0.0}


if __name__ == "__main__":
    import sys as _sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from recommender.build_embeddings import load_embeddings  # noqa: E402

    print("构建用户选课表 ...")
    uc = build_user_courses()
    print("加载已生成的课程嵌入 ...")
    emb, cids = load_embeddings()
    print(f"  已嵌入课程 {len(emb)} 门")
    print()
    rep = coverage_report(uc, emb)
    print(f"选课记录覆盖: {rep['covered_enrollments']:,}/{rep['total_enrollments']:,} "
          f"({rep['coverage']*100:.2f}%)")
    uv = build_user_vectors(uc, emb)
    if uv:
        anyu = next(iter(uv.values()))
        print(f"用户向量维度: {anyu.shape}")
        norms = np.linalg.norm(np.vstack(list(uv.values())), axis=1)
        print(f"用户向量范数: min={norms.min():.2f} max={norms.max():.2f}")
    else:
        print("暂无用户可构造画像（需要先跑更多课程图与嵌入）")
