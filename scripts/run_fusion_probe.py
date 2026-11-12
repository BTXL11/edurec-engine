"""融合原型探针：不确定性门控的表示融合（Beauty，冻结表示）。

用法
----
    python -m scripts.run_fusion_probe
    python -m scripts.run_fusion_probe --k 5 --neg 99 --out model/fusion_probe.json

这个脚本**不训练任何网络**，只验证融合机制本身是否成立：

1. ID 侧：直接取 STOSA 已训练好的物品分布
   ``item_mean_embeddings`` / ``item_cov_embeddings``（12103×64，行 0 为 padding）；
2. 多模态侧：自建 CLIP 图文特征（text/image 各 512 维）拼接，用岭回归映射到
   同一个 64 维行为空间（对应 M3CSR 的「每模态线性映射到行为空间」），
   映射残差作为该物品多模态表示的方差，并做尺度对齐；
3. 用户表示：最近 k 条历史的**精度加权**平均（只用 ID 侧，多模态不碰用户侧，
   避免把目标物品的信息漏进来）；
4. 不确定性：两种估计——
   ``trace``（序列分布协方差的迹，STOSA 原生口径）与
   ``dispersion``（历史物品均值之间的相邻欧氏距离，衡量兴趣漂移）；
5. 门控：验证集上学两参数 logistic（不确定性 → 多模态权重 g）；
6. 打分：对角高斯下的 2-Wasserstein 距离，1 vs 99 采样，报 HR@10 / NDCG@10 / MRR。

对照：纯 ID、纯多模态、固定 0.5、门控（两种不确定性），另做分桶增益与相关性诊断。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from math import erfc

import numpy as np

from engine.fusion.gate import (
    UncertaintyGate,
    sigmoid,
    summarize_uncertainty,
    wasserstein2,
)

DEFAULT_CKPT = (
    "STOSA/outputs/DistSAModel-Beauty-64-1-4-gelu-0.0-0.3-100-0.001-0.0-10-1.0-0.005.pt"
)
DEFAULT_FEATURES = "PRISM/data_local/Beauty"
CONFIGS = ("id_only", "mm_only", "fixed_0.5", "gated_trace", "gated_disp", "gated_tuned")


# --------------------------------------------------------------------- 数据
def load_sequences(path: str) -> list[np.ndarray]:
    """读取 ``user_id item item ...`` 形式的序列文件（item 从 1 开始编号）。"""
    seqs: list[np.ndarray] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:  # 至少两条交互才能留一评估
                continue
            seqs.append(np.asarray([int(x) for x in parts[1:]], dtype=np.int64))
    return seqs


def load_id_gaussians(ckpt_path: str) -> tuple[np.ndarray, np.ndarray]:
    """从 STOSA checkpoint 取物品高斯（对角）参数。"""
    import torch

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mean = state["item_mean_embeddings.weight"].to(torch.float64).numpy()
    var = state["item_cov_embeddings.weight"].to(torch.float64).numpy()
    var = np.clip(var, 1e-6, None)  # cov 权重训练时为正，仍兜底一次
    return mean, var


def load_modal_features(features_dir: str, n_items: int) -> np.ndarray:
    """拼接图文 CLIP 特征，返回 (n_items+1, 1024)，行 0 为 padding 全零。"""
    import torch

    text = torch.load(
        os.path.join(features_dir, "text_features_Beauty.pt"),
        map_location="cpu",
        weights_only=False,
    ).to(torch.float32).numpy()
    image = torch.load(
        os.path.join(features_dir, "image_features_Beauty.pt"),
        map_location="cpu",
        weights_only=False,
    ).to(torch.float32).numpy()
    if text.shape[0] != image.shape[0]:
        raise ValueError(f"图文特征行数不一致：{text.shape[0]} vs {image.shape[0]}")
    feats = np.zeros((n_items + 1, text.shape[1] + image.shape[1]), dtype=np.float64)
    n = min(text.shape[0], n_items)
    feats[1 : n + 1, : text.shape[1]] = text[:n]
    feats[1 : n + 1, text.shape[1] :] = image[:n]
    empty = int((np.abs(feats[1 : n + 1]).sum(axis=1) == 0).sum())
    print(f"[features] {feats.shape}，空行 {empty}/{n}（行号=物品 id，行 0 为 padding）")
    return feats


def fit_ridge_map(
    feats: np.ndarray,
    target: np.ndarray,
    train_ids: np.ndarray,
    lam: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """岭回归把多模态特征映射到 ID 行为空间，返回 (全部物品映射均值, 逐维残差方差)。"""
    ids = train_ids[(train_ids > 0) & (train_ids < feats.shape[0])]
    ids = np.unique(ids)
    ids = ids[np.abs(feats[ids]).sum(axis=1) > 0]
    X = feats[ids]
    Y = target[ids]
    gram = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float64)
    W = np.linalg.solve(gram, X.T @ Y)
    pred = feats @ W
    resid = (target[ids] - pred[ids]) ** 2
    var = np.full_like(target, float(np.mean(resid)), dtype=np.float64)
    var[ids] = resid
    print(
        f"[ridge] 训练物品 {ids.size}，特征维 {X.shape[1]}，"
        f"残差均值 {float(np.mean(resid)):.5f}"
    )
    return pred, np.clip(var, 1e-4, None)


# --------------------------------------------------------------- 用户表示
def sequence_repr(
    seq: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """最近 k 条交互的精度加权平均，方差按独立假设传播：Σ p_j² σ_j²。"""
    hist = seq[-k:] if seq.shape[0] > k else seq
    mu = mean[hist]
    sigma2 = var[hist]
    precision = 1.0 / (sigma2.mean(axis=1) + 1e-8)
    p = precision / precision.sum()
    user_mean = (p[:, None] * mu).sum(axis=0)
    user_var = ((p[:, None] ** 2) * sigma2).sum(axis=0)
    return user_mean, user_var


def interest_dispersion(seq: np.ndarray, mean: np.ndarray, k: int) -> float:
    """相邻历史物品均值的平均平方距离：衡量「兴趣漂移」而非「物品本身有多糊」。"""
    hist = seq[-k:] if seq.shape[0] > k else seq
    if hist.shape[0] < 2:
        return 0.0
    mu = mean[hist]
    step = np.diff(mu, axis=0)
    return float((step**2).sum(axis=1).mean())


# ------------------------------------------------------------------ 评估
def _mix(a: np.ndarray, b: np.ndarray, g: np.ndarray) -> np.ndarray:
    """按 (n,) 权重在两路物品表示上做凸组合，a/b 形状 (n, neg+1, d)。"""
    return (1.0 - g.reshape(-1, 1, 1)) * a + g.reshape(-1, 1, 1) * b


def build_candidates(
    rng: np.random.Generator, n_items: int, history: np.ndarray, neg: int
) -> np.ndarray:
    """采样 neg 个历史外的负样本。"""
    banned = set(int(x) for x in history)
    negs: list[int] = []
    while len(negs) < neg:
        j = int(rng.integers(1, n_items + 1))
        if j not in banned:
            negs.append(j)
    return np.asarray(negs, dtype=np.int64)


def evaluate(
    users: list[np.ndarray],
    mean_id: np.ndarray,
    var_id: np.ndarray,
    mean_mm: np.ndarray,
    var_mm: np.ndarray,
    k: int,
    neg: int,
    seed: int,
    weight_fn=None,
    chunk: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1 vs neg 排序，返回 (名次, 迹不确定性, 漂移不确定性)。

    ``weight_fn=None`` 表示纯 ID；否则按不确定性给出多模态权重 g∈[0,1]。
    候选集在所有对照之间共用同一个 seed，保证可比。
    """
    rng = np.random.default_rng(seed)
    ranks: list[np.ndarray] = []
    unc_trace: list[np.ndarray] = []
    unc_disp: list[np.ndarray] = []
    for start in range(0, len(users), chunk):
        block = users[start : start + chunk]
        cands = np.stack([build_candidates(rng, mean_id.shape[0] - 1, u, neg) for u in block])
        targets = np.asarray([u[-1] for u in block], dtype=np.int64)
        cands = np.concatenate([targets[:, None], cands], axis=1)  # 第 0 列是真值

        reps = [sequence_repr(u[:-1], mean_id, var_id, k) for u in block]
        user_mean = np.stack([r[0] for r in reps])
        user_var = np.stack([r[1] for r in reps])
        trace = summarize_uncertainty(user_var)
        disp = np.asarray([interest_dispersion(u[:-1], mean_id, k) for u in block])

        item_mean = mean_id[cands]
        item_var = var_id[cands]
        if weight_fn is not None:
            g = np.asarray(weight_fn(trace, disp), dtype=np.float64).reshape(-1)
            item_mean = _mix(item_mean, mean_mm[cands], g)
            item_var = _mix(item_var, var_mm[cands], g)

        dist = wasserstein2(
            user_mean[:, None, :], user_var[:, None, :], item_mean, item_var
        )
        true_score = dist[:, 0]
        ranks.append(1 + (dist < true_score[:, None]).sum(axis=1))
        unc_trace.append(trace)
        unc_disp.append(disp)
    return np.concatenate(ranks), np.concatenate(unc_trace), np.concatenate(unc_disp)


def metrics(ranks: np.ndarray) -> dict[str, float]:
    """HR@10 / NDCG@10 / MRR（1 vs 99 采样口径）。"""
    ranks = np.asarray(ranks, dtype=np.float64)
    hit = ranks <= 10
    gains = np.where(hit, 1.0 / np.log2(ranks + 1), 0.0)
    return {
        "hr@10": float(hit.mean()),
        "ndcg@10": float(gains.mean()),
        "mrr": float((1.0 / ranks).mean()),
    }


def paired_stats(
    rank_base: np.ndarray,
    rank_new: np.ndarray,
    *,
    k: int = 10,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, float | int | list[float]]:
    """两个配置在同一批用户上的配对检验：McNemar（命中与否）+ 自助法置信区间。"""
    hit_base = rank_base <= k
    hit_new = rank_new <= k
    wins = int((~hit_base & hit_new).sum())   # 基准没中、新配置中了
    losses = int((hit_base & ~hit_new).sum())  # 基准中了、新配置没中
    total = wins + losses
    chi2 = (abs(wins - losses) - 1) ** 2 / total if total else 0.0
    p_value = float(erfc(np.sqrt(max(chi2, 0.0) / 2.0)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, hit_base.size, size=(n_boot, hit_base.size))
    deltas = hit_new[idx].mean(axis=1) - hit_base[idx].mean(axis=1)
    return {
        "hr@10_base": float(hit_base.mean()),
        "hr@10_new": float(hit_new.mean()),
        "delta": float(hit_new.mean() - hit_base.mean()),
        "wins": wins,
        "losses": losses,
        "chi2": float(chi2),
        "p_value": p_value,
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
    }


def bucket_report(
    name: str, unc: np.ndarray, rank_sets: dict[str, np.ndarray], weights: dict[str, np.ndarray]
) -> list[dict]:
    """按不确定性四分桶看各配置的 HR@10，检查增益是否随不确定性变化。"""
    quartiles = np.quantile(unc, [0.25, 0.5, 0.75])
    buckets = np.digitize(unc, quartiles)
    out = []
    for b in range(4):
        mask = buckets == b
        if not mask.any():
            continue
        row = {
            "uncertainty": name,
            "bucket": b,
            "n": int(mask.sum()),
            "mean_unc": float(unc[mask].mean()),
            "mean_gate": float(weights[name][mask].mean()) if name in weights else None,
            "hr10_id": float((rank_sets["id_only"][mask] <= 10).mean()),
            "hr10_fixed": float((rank_sets["fixed_0.5"][mask] <= 10).mean()),
            "hr10_gated": float((rank_sets[f"gated_{name}"][mask] <= 10).mean()),
        }
        row["delta"] = row["hr10_gated"] - row["hr10_fixed"]
        out.append(row)
    return out


# ------------------------------------------------------------------ 主流程
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="不确定性门控融合原型探针")
    ap.add_argument("--seq", default="STOSA/data/Beauty.txt")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--k", type=int, default=5, help="历史窗口长度")
    ap.add_argument("--neg", type=int, default=99, help="负样本数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-users", type=int, default=0, help="只跑前 N 个用户（0=全部）")
    ap.add_argument("--out", default="model/fusion_probe.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    seqs = load_sequences(args.seq)
    if args.max_users > 0:
        seqs = seqs[: args.max_users]
    mean_id, var_id = load_id_gaussians(args.ckpt)
    n_items = mean_id.shape[0] - 1
    feats = load_modal_features(args.features, n_items)
    print(f"[data] 用户 {len(seqs)}，物品 {n_items}，加载 {time.time() - t0:.1f}s")

    train_ids = np.unique(np.concatenate([u[:-2] for u in seqs]))
    mean_mm, var_mm = fit_ridge_map(feats, mean_id, train_ids)
    # 尺度对齐：两路方差若不在同一量级，W2 的方差项会单方面主导距离
    ref = float(np.median(summarize_uncertainty(var_id[train_ids])))
    cur = float(np.median(summarize_uncertainty(var_mm[train_ids])))
    scale = ref / max(cur, 1e-8)
    var_mm = np.clip(var_mm * scale, 1e-6, None)
    print(f"[calib] 方差尺度 ID={ref:.4f} 多模态(原始)={cur:.4f} 缩放={scale:.3f}")

    # ---- 验证：倒数第二条为目标，学两个门控
    valid_users = [u for u in seqs if u.shape[0] >= 3]
    v_id, v_trace, v_disp = evaluate(
        valid_users, mean_id, var_id, mean_mm, var_mm, args.k, args.neg, args.seed
    )
    v_mm, _, _ = evaluate(
        valid_users, mean_id, var_id, mean_mm, var_mm, args.k, args.neg, args.seed,
        weight_fn=lambda t, d: np.ones_like(t),
    )
    label = (v_mm < v_id).astype(np.float64)
    gate_trace = UncertaintyGate().fit(v_trace, label)
    gate_disp = UncertaintyGate().fit(v_disp, label)
    print(
        f"[gate:trace] slope={gate_trace.slope:+.3f} b={gate_trace.intercept:+.3f} "
        f"conv={gate_trace.converged}"
    )
    print(
        f"[gate:disp ] slope={gate_disp.slope:+.3f} b={gate_disp.intercept:+.3f} "
        f"conv={gate_disp.converged}  多模态胜率={label.mean():.3f}"
    )

    # ---- 斜率调参：logistic 拟合出的斜率天然偏平（权重被压在 0.4~0.55），
    #      在验证集上扫一遍斜率，看「更陡的门控」是否真的更好
    mean_d, std_d = float(v_disp.mean()), float(v_disp.std() or 1.0)

    def _slope_fn(a: float, b: float):
        return lambda _t, d: sigmoid(a * (d - mean_d) / max(std_d, 1e-8) + b)

    slope_grid = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
    best_slope, best_hr = 0.0, -1.0
    for a in slope_grid:
        ranks, _, _ = evaluate(
            valid_users, mean_id, var_id, mean_mm, var_mm, args.k, args.neg, args.seed,
            weight_fn=_slope_fn(a, gate_disp.intercept),
        )
        hr = float((ranks <= 10).mean())
        if hr > best_hr:
            best_slope, best_hr = a, hr
    gate_tuned = UncertaintyGate(
        slope=best_slope,
        intercept=gate_disp.intercept,
        mean_unc=mean_d,
        std_unc=std_d,
    )
    print(f"[gate:tuned] slope={best_slope:+.2f}（网格搜索，验证 HR@10={best_hr:.4f}）")

    # ---- 测试：最后一条为目标；所有对照共用候选集
    rank_sets: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    test_trace = test_disp = None
    for name, fn in [
        ("id_only", None),
        ("mm_only", lambda t, d: np.ones_like(t)),
        ("fixed_0.5", lambda t, d: np.full_like(t, 0.5)),
        ("gated_trace", lambda t, d: gate_trace.weights(t)),
        ("gated_disp", lambda t, d: gate_disp.weights(d)),
        ("gated_tuned", lambda t, d: gate_tuned.weights(d)),
    ]:
        ranks, tr, dp = evaluate(
            seqs, mean_id, var_id, mean_mm, var_mm, args.k, args.neg, args.seed + 100,
            weight_fn=fn,
        )
        rank_sets[name] = ranks
        test_trace, test_disp = tr, dp
        if name.startswith("gated"):
            key = name.split("_", 1)[1]
            if key == "trace":
                weights[key] = gate_trace.weights(tr)
            elif key == "tuned":
                weights[key] = gate_tuned.weights(dp)
            else:
                weights[key] = gate_disp.weights(dp)
    results = {name: metrics(rank_sets[name]) for name in CONFIGS}
    for name in CONFIGS:
        m = results[name]
        print(
            f"[test] {name:12s} HR@10={m['hr@10']:.4f} "
            f"NDCG@10={m['ndcg@10']:.4f} MRR={m['mrr']:.4f}"
        )

    # ---- 配对检验：小差距必须过显著性，否则不能当结论
    significance = {
        "fixed_vs_id": paired_stats(rank_sets["id_only"], rank_sets["fixed_0.5"]),
        "tuned_vs_fixed": paired_stats(rank_sets["fixed_0.5"], rank_sets["gated_tuned"]),
        "tuned_vs_id": paired_stats(rank_sets["id_only"], rank_sets["gated_tuned"]),
        "tracegate_vs_fixed": paired_stats(rank_sets["fixed_0.5"], rank_sets["gated_trace"]),
    }
    for key, stat in significance.items():
        print(
            f"[sig] {key:20s} ΔHR@10={stat['delta']:+.4f} "
            f"95%CI=[{stat['ci95'][0]:+.4f},{stat['ci95'][1]:+.4f}] "
            f"wins/losses={stat['wins']}/{stat['losses']} p={stat['p_value']:.3g}"
        )

    # ---- 分桶诊断
    buckets = bucket_report("trace", test_trace, rank_sets, weights)
    buckets += bucket_report("disp", test_disp, rank_sets, weights)
    buckets += bucket_report("tuned", test_disp, rank_sets, weights)
    for row in buckets:
        print(
            f"[bucket {row['uncertainty']:5s} {row['bucket']}] n={row['n']:6d} "
            f"unc={row['mean_unc']:.3f} g={row['mean_gate']:.3f} "
            f"HR@10 id={row['hr10_id']:.4f} fixed={row['hr10_fixed']:.4f} "
            f"gated={row['hr10_gated']:.4f} Δ={row['delta']:+.4f}"
        )

    # ---- 相关性诊断：不确定性到底在度量什么
    pop = np.zeros(n_items + 1, dtype=np.float64)
    for u in seqs:
        np.add.at(pop, u, 1.0)
    last_items = np.asarray([u[-1] for u in seqs])
    hist_len = np.asarray([u.shape[0] - 1 for u in seqs], dtype=np.float64)
    diag = {
        "corr_trace_vs_histlen": float(np.corrcoef(test_trace, hist_len)[0, 1]),
        "corr_trace_vs_last_item_pop": float(np.corrcoef(test_trace, pop[last_items])[0, 1]),
        "corr_disp_vs_histlen": float(np.corrcoef(test_disp, hist_len)[0, 1]),
        "corr_disp_vs_last_item_pop": float(np.corrcoef(test_disp, pop[last_items])[0, 1]),
        "corr_trace_vs_disp": float(np.corrcoef(test_trace, test_disp)[0, 1]),
        "hr10_trace_quartile4_vs_1": float(
            (rank_sets["id_only"][test_trace > np.quantile(test_trace, 0.75)] <= 10).mean()
            - (rank_sets["id_only"][test_trace <= np.quantile(test_trace, 0.25)] <= 10).mean()
        ),
    }
    for k, v in diag.items():
        print(f"[diag] {k} = {v:+.4f}")

    payload = {
        "config": vars(args),
        "gates": {
            "trace": gate_trace.as_dict(),
            "disp": gate_disp.as_dict(),
            "tuned": gate_tuned.as_dict(),
        },
        "results": results,
        "buckets": buckets,
        "significance": significance,
        "diagnostics": diag,
        "mean_uncertainty": {
            "trace": float(test_trace.mean()),
            "disp": float(test_disp.mean()),
        },
        "multimodal_win_rate_valid": float(label.mean()),
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[done] {args.out}（{payload['elapsed_s']}s）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
