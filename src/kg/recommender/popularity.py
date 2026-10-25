"""从真实观看行为里算课程热度。

数据源：data/MOOCCube/additional_information/user_video_act.json（JSON Lines，1.7 GB）
每条记录是一个用户及其 activity 列表，activity 字段：
    course_id, video_id, watching_count, video_duration,
    local_watching_time, video_progress_time, ...

本项目原本要接"B 站播放/点赞/收藏/投币"，但本机没有 B 站数据；MOOCCube 自带的
真实观看行为是等价可用的热度来源。三个信号：

    unique_learners  独立学习者数（去重用户）
    total_watching   总观看次数（watching_count 求和）
    mean_completion  平均观看完成度 mean(min(local_watching_time / video_duration, 1))

三者各自归一化后加权求和，得到 hotness ∈ [0, 1]。
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.stats import DATA, load_cache, save_cache  # noqa: E402

CACHE_NAME = "popularity.json"


def compute_popularity(verbose=True, limit=None):
    """流式扫描观看行为，聚合出每个课程的原始热度指标。

    limit: 只处理前 N 个用户（调试用）；None 表示全量。
    """
    path = os.path.join(DATA, "additional_information", "user_video_act.json")
    import json

    learners_by_course = {}
    stats = {}

    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = rec.get("id")
            for act in rec.get("activity", []):
                cid = act.get("course_id")
                if not cid:
                    continue
                s = stats.get(cid)
                if s is None:
                    s = stats[cid] = {
                        "unique_learners": 0,
                        "total_watching": 0,
                        "completion_sum": 0.0,
                        "completion_n": 0,
                    }
                    learners_by_course[cid] = set()
                if uid not in learners_by_course[cid]:
                    learners_by_course[cid].add(uid)
                    s["unique_learners"] += 1

                wc = act.get("watching_count") or 0
                s["total_watching"] += int(wc)

                dur = act.get("video_duration") or 0
                watched = act.get("local_watching_time") or 0
                if dur and dur > 0 and watched:
                    s["completion_sum"] += min(float(watched) / float(dur), 1.0)
                    s["completion_n"] += 1

            if verbose and (i + 1) % 10000 == 0:
                print(f"  已处理 {i+1} 个用户, {len(stats)} 门课程, "
                      f"{time.time()-t0:.0f}s", flush=True)

    for cid, s in stats.items():
        n = s.pop("completion_n")
        s["mean_completion"] = (s.pop("completion_sum") / n) if n else 0.0

    if verbose:
        print(f"  扫描完成: {len(stats)} 门课程, 用时 {time.time()-t0:.0f}s")
    return stats


def normalize_popularity(stats, weights=(0.5, 0.2, 0.3)):
    """把原始指标归一化成 hotness ∈ [0,1]。

    用 log1p 压缩长尾（热门课程动辄数万学习者，线性归一化会把其余全压到 0 附近），
    每个信号除以自身最大值。
    """
    w_learners, w_watch, w_completion = weights
    if not stats:
        return {}

    def lognorm(values):
        logged = [math.log1p(v) for v in values]
        m = max(logged) if logged else 0.0
        return [v / m if m > 0 else 0.0 for v in logged]

    cids = list(stats.keys())
    ln = lognorm([stats[c]["unique_learners"] for c in cids])
    lw = lognorm([stats[c]["total_watching"] for c in cids])
    lc = lognorm([stats[c]["mean_completion"] for c in cids])

    out = {}
    for cid, a, b, c in zip(cids, ln, lw, lc):
        hot = w_learners * a + w_watch * b + w_completion * c
        out[cid] = {
            **stats[cid],
            "hotness": round(hot, 6),
            "norm_learners": round(a, 6),
            "norm_watching": round(b, 6),
            "norm_completion": round(c, 6),
        }
    return out


def get_popularity(force=False, verbose=True):
    """带缓存的入口：第一次扫描后写入 rec_cache/popularity.json。"""
    if not force:
        cached = load_cache(CACHE_NAME)
        if cached:
            if verbose:
                print(f"  使用缓存热度数据（{len(cached)} 门课程）")
            return cached
    raw = compute_popularity(verbose=verbose)
    norm = normalize_popularity(raw)
    save_cache(CACHE_NAME, norm)
    if verbose:
        print(f"  已缓存到 rec_cache/{CACHE_NAME}")
    return norm


if __name__ == "__main__":
    import sys as _sys

    limit = None
    if len(_sys.argv) > 1 and _sys.argv[1].isdigit():
        limit = int(_sys.argv[1])
    print(f"开始扫描观看行为{f'（前 {limit} 个用户）' if limit else '（全量）'} ...")
    raw = compute_popularity(limit=limit)
    norm = normalize_popularity(raw)
    save_cache(CACHE_NAME, norm)
    print(f"\n共 {len(norm)} 门课程有观看记录\n")

    top = sorted(norm.items(), key=lambda kv: -kv[1]["hotness"])[:15]
    print(f"{'课程':52s} {'学习者':>7s} {'观看次数':>9s} {'完成度':>7s} {'热度':>7s}")
    print("-" * 92)
    for cid, s in top:
        short = cid if len(cid) <= 50 else cid[:47] + "..."
        print(f"{short:52s} {s['unique_learners']:7d} {s['total_watching']:9d} "
              f"{s['mean_completion']:7.3f} {s['hotness']:7.3f}")
    save_cache(CACHE_NAME, norm)
    print(f"\n缓存已写入 rec_cache/{CACHE_NAME}")
