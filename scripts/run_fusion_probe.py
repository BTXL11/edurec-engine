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
   避免把目标物品的信息漏进来）；``--backbone stosa`` 时改为调用 STOSA 的
   ``DistSAModel`` 编码整段历史并取最后一个位置，与上游评估口径一致；
4. 不确定性：两种估计——
   ``trace``（序列分布协方差的迹，STOSA 原生口径）与
   ``dispersion``（历史物品均值之间的相邻欧氏距离，衡量兴趣漂移）；
5. 门控：验证集上学 logistic（不确定性 → 多模态权重 g），并在验证集上联合搜
   ``(floor, slope)`` 得到「自适应图文」：迷茫的用户多信课程内容，目标明确的用户
   多信历史，但内容始终保留 ``floor`` 的下限；
6. 打分：对角高斯下的 2-Wasserstein 距离，1 vs 99 采样，报 HR@10 / NDCG@10 / MRR。

对照：纯 ID、纯多模态、固定 0.5、门控（两种不确定性）、自适应图文，另做分桶增益、
「迷茫→内容」前提检验与相关性诊断。
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
CONFIGS = (
    "id_only",
    "mm_only",
    "fixed_0.5",
    "gated_trace",
    "gated_disp",
    "gated_adaptive",
)


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


def load_id_gaussians(
    ckpt_path: str, item_cov: str = "elu"
) -> tuple[np.ndarray, np.ndarray]:
    """从 STOSA checkpoint 取物品高斯（对角）参数。

    ``item_cov="elu"`` 走上游评估口径 ``elu(cov) + 1``（``trainers.py:463``），
    ``"raw"`` 保留早期实验直接取权重、负值裁到 1e-6 的做法。
    """
    import torch

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mean = state["item_mean_embeddings.weight"].to(torch.float64).numpy()
    var = state["item_cov_embeddings.weight"].to(torch.float64).numpy()
    if item_cov == "elu":
        var = np.maximum(var, 0.0) + 1.0
    else:
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
    folds: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """岭回归把多模态特征映射到 ID 行为空间，返回 (全部物品映射均值, 逐维残差方差)。

    ``folds >= 2`` 时对训练物品做 K 折交叉拟合：被评估的物品只由没见过它的
    模型给出向量，避免「在参与评估的物品上拟合、再在同样物品上评估」的记忆型
    泄漏（留出物品 R² 为负时，样本内拟合会把噪声当信号）。
    """
    ids = train_ids[(train_ids > 0) & (train_ids < feats.shape[0])]
    ids = np.unique(ids)
    ids = ids[np.abs(feats[ids]).sum(axis=1) > 0]
    X = feats[ids]
    Y = target[ids]

    def _solve(Xk: np.ndarray, Yk: np.ndarray) -> np.ndarray:
        gram = Xk.T @ Xk + lam * np.eye(Xk.shape[1], dtype=np.float64)
        return np.linalg.solve(gram, Xk.T @ Yk)

    pred = feats @ _solve(X, Y)
    resid_dim = (target[ids] - pred[ids]) ** 2
    resid = resid_dim.mean(axis=1)

    if folds >= 2:
        rng = np.random.default_rng(0)
        parts = np.array_split(rng.permutation(ids.size), folds)
        oof = np.zeros((ids.size, target.shape[1]), dtype=np.float64)
        for part in parts:
            keep = np.setdiff1d(np.arange(ids.size), part)
            oof[part] = feats[ids[part]] @ _solve(X[keep], Y[keep])
            resid_dim[part] = (Y[part] - oof[part]) ** 2
        resid = resid_dim.mean(axis=1)
        pred[ids] = oof
        ss_res = ((Y - oof) ** 2).sum()
        ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum()
        print(
            f"[ridge] {folds} 折交叉拟合：留出物品 R² = {1 - ss_res / ss_tot:+.4f}"
        )

    var = np.full_like(target, float(np.mean(resid_dim)), dtype=np.float64)
    var[ids] = resid_dim
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
    rng: np.random.Generator,
    n_items: int,
    history: np.ndarray,
    neg: int,
    pool: np.ndarray | None = None,
) -> np.ndarray:
    """采样 neg 个历史外的负样本；``pool`` 给定时只从该物品池里抽（冷启动协议）。"""
    banned = set(int(x) for x in history)
    if pool is None:
        allowed = np.arange(1, n_items + 1, dtype=np.int64)
    else:
        allowed = np.unique(np.asarray(pool, dtype=np.int64))
    if banned:
        allowed = allowed[~np.isin(allowed, list(banned))]
    if allowed.size == 0:
        raise ValueError("负样本池里没有可用物品（都被历史占用了）")
    # 池子够大就不放回地抽，保证负样本互不重复；不够大时允许重复，避免死循环。
    replace = allowed.size < neg
    return np.asarray(rng.choice(allowed, size=neg, replace=replace), dtype=np.int64)


def evaluate_grid(
    users: list[np.ndarray],
    mean_id: np.ndarray,
    var_id: np.ndarray,
    mean_mm: np.ndarray,
    var_mm: np.ndarray,
    k: int,
    neg: int,
    seed: int,
    weight_fns: dict[str, "callable | None"],
    encoder=None,
    pool: np.ndarray | None = None,
    chunk: int = 2000,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """一次遍历候选集，同时评估多个融合配置。

    ``weight_fns`` 的键是配置名，值为 ``fn(trace, disp) -> g``；值为 ``None``
    表示纯 ID。候选集在所有配置之间共用，因此各配置的名次可逐用户配对比较。

    ``encoder`` 为 None 时用户表示走「最近 k 条精度加权平均」；否则调用
    ``encoder(histories) -> (均值, 方差)``（如 STOSA 序列编码器）。

    返回 ``(ranks_by_config, trace, disp)``。
    """
    rng = np.random.default_rng(seed)
    ranks: dict[str, list[np.ndarray]] = {name: [] for name in weight_fns}
    unc_trace: list[np.ndarray] = []
    unc_disp: list[np.ndarray] = []
    for start in range(0, len(users), chunk):
        block = users[start : start + chunk]
        cands = np.stack(
            [build_candidates(rng, mean_id.shape[0] - 1, u, neg, pool) for u in block]
        )
        targets = np.asarray([u[-1] for u in block], dtype=np.int64)
        cands = np.concatenate([targets[:, None], cands], axis=1)  # 第 0 列是真值

        if encoder is None:
            reps = [sequence_repr(u[:-1], mean_id, var_id, k) for u in block]
        else:
            rep_mean, rep_var = encoder([u[:-1] for u in block])
            reps = list(zip(rep_mean, rep_var))
        user_mean = np.stack([r[0] for r in reps])
        user_var = np.stack([r[1] for r in reps])
        trace = summarize_uncertainty(user_var)
        disp = np.asarray([interest_dispersion(u[:-1], mean_id, k) for u in block])

        item_mean_id = mean_id[cands]
        item_var_id = var_id[cands]
        item_mean_mm = mean_mm[cands]
        item_var_mm = var_mm[cands]

        for name, weight_fn in weight_fns.items():
            if weight_fn is None:
                item_mean, item_var = item_mean_id, item_var_id
            else:
                g = np.asarray(weight_fn(trace, disp), dtype=np.float64).reshape(-1)
                item_mean = _mix(item_mean_id, item_mean_mm, g)
                item_var = _mix(item_var_id, item_var_mm, g)
            dist = wasserstein2(
                user_mean[:, None, :], user_var[:, None, :], item_mean, item_var
            )
            true_score = dist[:, 0]
            ranks[name].append(1 + (dist < true_score[:, None]).sum(axis=1))

        unc_trace.append(trace)
        unc_disp.append(disp)
    return (
        {name: np.concatenate(chunks) for name, chunks in ranks.items()},
        np.concatenate(unc_trace),
        np.concatenate(unc_disp),
    )


def evaluate_rank_fusion(
    users: list[np.ndarray],
    mean_id: np.ndarray,
    var_id: np.ndarray,
    feats: np.ndarray,
    k: int,
    neg: int,
    seed: int,
    weight_fns: dict[str, "callable | None"],
    encoder=None,
    pool: np.ndarray | None = None,
    chunk: int = 2000,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """分数级融合：ID 走 W2 距离、图文走内容向量余弦，两路各自转成名次分再加权。

    这样不需要把 CLIP 特征线性映射进行为空间（交叉拟合后留出物品 R² 为负），
    也绕开了两路量纲不一致的问题。``weight_fns[key]`` 返回内容侧权重 g，
    ``None`` 表示纯 ID；各配置共用候选集，名次可逐用户配对比较。
    """
    rng = np.random.default_rng(seed)
    # CLIP 特征有一个很大的公共方向（所有物品的平均向量），不减掉的话余弦
    # 主要反映「谁更像平均物品」，内容个性被淹没。这里用全库均值做中心化。
    n_rows = feats.shape[0]
    center = feats[1:n_rows].mean(axis=0, keepdims=True)
    centered = feats.copy()
    centered[1:n_rows] -= center
    feat_norm = centered / np.maximum(
        np.linalg.norm(centered, axis=1, keepdims=True), 1e-12
    )
    ranks = {name: np.empty(len(users), dtype=np.int64) for name in weight_fns}
    traces = np.empty(len(users), dtype=np.float64)
    disps = np.empty(len(users), dtype=np.float64)

    for start in range(0, len(users), chunk):
        block = users[start : start + chunk]
        targets = np.asarray([u[-1] for u in block], dtype=np.int64)
        cands = np.stack(
            [build_candidates(rng, mean_id.shape[0] - 1, u, neg, pool) for u in block]
        )
        cands = np.concatenate([targets[:, None], cands], axis=1)

        hist = [u[:-1] for u in block]
        if encoder is None:
            reps = [sequence_repr(h, mean_id, var_id, k) for h in hist]
        else:
            rep_mean, rep_var = encoder(hist)
            reps = list(zip(rep_mean, rep_var))
        user_mean = np.stack([r[0] for r in reps])
        user_var = np.stack([r[1] for r in reps])

        trace = summarize_uncertainty(user_var)
        disp = np.asarray([interest_dispersion(h, mean_id, k) for h in hist])

        content = np.stack([feat_norm[h[-k:]].mean(axis=0) for h in hist])
        content /= np.maximum(np.linalg.norm(content, axis=1, keepdims=True), 1e-12)

        for i in range(len(block)):
            dist = wasserstein2(
                user_mean[i][None, :],
                user_var[i][None, :],
                mean_id[cands[i]],
                var_id[cands[i]],
            )
            sim = feat_norm[cands[i]] @ content[i]
            n = cands.shape[1]
            pos_id = np.empty(n)
            pos_id[np.argsort(dist)] = np.arange(n)
            pos_mm = np.empty(n)
            pos_mm[np.argsort(-sim)] = np.arange(n)
            score_id = 1.0 - pos_id / max(n - 1, 1)
            score_mm = 1.0 - pos_mm / max(n - 1, 1)

            t_i = np.array([trace[i]])
            d_i = np.array([disp[i]])
            for name, fn in weight_fns.items():
                if fn is None:
                    rank = 1 + int((dist < dist[0]).sum())
                else:
                    g = float(np.asarray(fn(t_i, d_i), dtype=np.float64).reshape(-1)[0])
                    score = (1.0 - g) * score_id + g * score_mm
                    rank = 1 + int((score > score[0]).sum())
                ranks[name][start + i] = rank

        traces[start : start + len(block)] = trace
        disps[start : start + len(block)] = disp

    return ranks, traces, disps


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
    """单配置包装：``weight_fn=None`` 表示纯 ID。"""
    ranks, trace, disp = evaluate_grid(
        users, mean_id, var_id, mean_mm, var_mm, k, neg, seed,
        {"main": weight_fn}, chunk,
    )
    return ranks["main"], trace, disp


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
    name: str,
    unc: np.ndarray,
    rank_sets: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    gated_key: str,
) -> list[dict]:
    """按不确定性四分桶看各配置的 HR@10，检查增益是否随不确定性变化。

    ``gated_key`` 是 ``rank_sets`` 里的门控配置名（如 ``gated_adaptive``），
    它的权重从 ``weights[gated_key.removeprefix('gated_')]`` 取。
    """
    quartiles = np.quantile(unc, [0.25, 0.5, 0.75])
    buckets = np.digitize(unc, quartiles)
    gate_weights = weights.get(gated_key.removeprefix("gated_"))
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
            "mean_gate": float(gate_weights[mask].mean()) if gate_weights is not None else None,
            "hr10_id": float((rank_sets["id_only"][mask] <= 10).mean()),
            "hr10_mm": float((rank_sets["mm_only"][mask] <= 10).mean()),
            "hr10_fixed": float((rank_sets["fixed_0.5"][mask] <= 10).mean()),
            "hr10_gated": float((rank_sets[gated_key][mask] <= 10).mean()),
        }
        row["delta"] = row["hr10_gated"] - row["hr10_fixed"]
        row["delta_vs_id"] = row["hr10_gated"] - row["hr10_id"]
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
    ap.add_argument(
        "--cold-threshold",
        type=int,
        default=-1,
        help="冷启动协议：只用训练段出现次数 <= N 的物品做候选池（-1=关闭）",
    )
    ap.add_argument(
        "--fusion",
        choices=("embedding", "score"),
        default="score",
        help="融合层级：score=两路名次分加权（默认），embedding=物品表示凸组合",
    )
    ap.add_argument(
        "--backbone",
        choices=("avg", "stosa"),
        default="avg",
        help="用户表示来源：avg=最近 k 条精度加权平均，stosa=DistSAModel 序列编码器",
    )
    ap.add_argument("--stosa-dir", default="STOSA", help="vendored STOSA 目录")
    ap.add_argument(
        "--ridge-folds",
        type=int,
        default=5,
        help="岭回归交叉拟合折数（>=2 时启用，避免多模态映射在评估物品上记忆）",
    )
    ap.add_argument(
        "--align-user",
        choices=("none", "zscore"),
        default="none",
        help="用户均值与物品均值是否按训练物品做逐维标准化（编码器输出与物品均值不同量纲时用）",
    )
    ap.add_argument(
        "--item-cov",
        choices=("elu", "raw"),
        default="elu",
        help="物品方差口径：elu=上游评估的 elu(cov)+1，raw=直接取权重",
    )
    args = ap.parse_args(argv)

    t0 = time.time()
    seqs = load_sequences(args.seq)
    if args.max_users > 0:
        seqs = seqs[: args.max_users]
    mean_id, var_id = load_id_gaussians(args.ckpt, args.item_cov)
    n_items = mean_id.shape[0] - 1
    feats = load_modal_features(args.features, n_items)
    print(f"[data] 用户 {len(seqs)}，物品 {n_items}，加载 {time.time() - t0:.1f}s")

    if args.backbone == "stosa":
        from engine.fusion.stosa_backbone import load_backbone

        backbone = load_backbone(args.ckpt, args.stosa_dir)
        encode = backbone.encode
        print(
            f"[backbone] STOSA DistSAModel d={backbone.hidden_size} "
            f"len={backbone.max_seq_length}，用户侧改用序列编码器"
        )
    else:
        encode = None

    train_ids = np.unique(np.concatenate([u[:-2] for u in seqs]))

    pool: np.ndarray | None = None
    pool_set: set[int] | None = None
    if args.cold_threshold >= 0:
        counts = np.bincount(
            np.concatenate([u[:-1] for u in seqs]), minlength=n_items + 1
        )
        pool = np.flatnonzero(counts <= args.cold_threshold)
        pool = pool[pool > 0]
        pool_set = set(int(x) for x in pool)
        before = len(seqs)
        seqs = [u for u in seqs if int(u[-1]) in pool_set]
        print(
            f"[cold] 阈值 <= {args.cold_threshold}：物品池 {pool.size}（{pool.size / n_items:.1%}），"
            f"末位物品在池内的用户 {len(seqs)}/{before}（{len(seqs) / before:.1%}）"
        )

    if args.align_user == "zscore":
        # 真编码器的用户均值与物品均值不同量纲（物品均值是嵌入表一行，
        # 用户均值是最后一层注意力的输出），不归一化会让 W2 的方差项 /
        # 均值项单方面主导。这里用训练物品的逐维均值与标准差同时作用到
        # 物品侧与用户侧，保证两路距离在同一坐标系里可比。
        _mu = mean_id[train_ids].mean(axis=0)
        _sd = mean_id[train_ids].std(axis=0) + 1e-6
        mean_id = (mean_id - _mu) / _sd
        var_id = np.maximum(var_id / (_sd ** 2), 1e-8)
        if encode is not None:
            _raw_encode = encode

            def encode(histories):  # type: ignore[misc]
                _m, _v = _raw_encode(histories)
                return (_m - _mu) / _sd, np.maximum(_v / (_sd ** 2), 1e-8)

        print(
            f"[align] 用户/物品均值逐维 z-score（{_mu.shape[0]} 维，"
            f"方差按 sd² 缩放），作用于 id 与编码器两侧"
        )

    if args.fusion == "embedding":
        mean_mm, var_mm = fit_ridge_map(
            feats, mean_id, train_ids, folds=args.ridge_folds
        )
        # 尺度对齐：两路方差若不在同一量级，W2 的方差项会单方面主导距离
        ref = float(np.median(summarize_uncertainty(var_id[train_ids])))
        cur = float(np.median(summarize_uncertainty(var_mm[train_ids])))
        scale = ref / max(cur, 1e-8)
        var_mm = np.clip(var_mm * scale, 1e-6, None)
        print(f"[calib] 方差尺度 ID={ref:.4f} 多模态(原始)={cur:.4f} 缩放={scale:.3f}")
    else:
        mean_mm = var_mm = np.zeros((1, mean_id.shape[1]), dtype=np.float64)
        print("[fusion] 分数级融合：跳过 CLIP→行为空间映射与方差标定")

    def _evaluate(users, weight_fns, eval_seed):
        if args.fusion == "score":
            return evaluate_rank_fusion(
                users,
                mean_id,
                var_id,
                feats,
                args.k,
                args.neg,
                eval_seed,
                weight_fns,
                encoder=encode,
                pool=pool,
            )
        return evaluate_grid(
            users,
            mean_id,
            var_id,
            mean_mm,
            var_mm,
            args.k,
            args.neg,
            eval_seed,
            weight_fns,
            encoder=encode,
            pool=pool,
        )

    # ---- 验证：倒数第二条为目标，一次遍历同时得到纯 ID 与纯多模态名次
    valid_users = [
        u[:-1]
        for u in seqs
        if u.shape[0] >= 3 and (pool_set is None or int(u[-2]) in pool_set)
    ]
    v_ranks, v_trace, v_disp = _evaluate(
        valid_users,
        {"id": None, "mm": lambda t, d: np.ones_like(t)},
        args.seed,
    )
    v_id, v_mm = v_ranks["id"], v_ranks["mm"]
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

    # ---- 自适应图文：在验证集上联合搜 (floor, slope)
    #      这是「学生越迷茫越看课程本身、越明确越信历史」的可证伪版本：
    #      * 最优 floor≈0 且 slope>0  -> 内容只在迷茫时补位（强假设成立）；
    #      * 最优 floor>0             -> 目标明确的用户也该保留一部分内容信号（弱假设）。
    mean_d, std_d = float(v_disp.mean()), float(v_disp.std() or 1.0)

    def _gate_fn(floor: float, a: float, b: float):
        return lambda _t, d: floor + (1.0 - floor) * sigmoid(
            a * (d - mean_d) / max(std_d, 1e-8) + b
        )

    floor_grid = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    slope_grid = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
    gate_candidates = {
        f"f{floor:.1f}_s{a:.1f}": _gate_fn(floor, a, gate_disp.intercept)
        for floor in floor_grid
        for a in slope_grid
    }
    cand_ranks, _, _ = _evaluate(valid_users, gate_candidates, args.seed)
    best_name, best_hr = max(
        ((name, float((ranks <= 10).mean())) for name, ranks in cand_ranks.items()),
        key=lambda item: item[1],
    )
    best_floor = float(best_name.split("_")[0][1:])
    best_slope = float(best_name.split("_s")[1])
    gate_adaptive = UncertaintyGate(
        slope=best_slope,
        intercept=gate_disp.intercept,
        mean_unc=mean_d,
        std_unc=std_d,
        floor=best_floor,
    )
    print(
        f"[gate:adaptive] floor={best_floor:.2f} slope={best_slope:+.2f} "
        f"(验证 HR@10={best_hr:.4f})"
    )

    # ---- 测试：最后一条为目标；所有对照共用候选集，一次遍历全部配置
    test_fns = {
        "id_only": None,
        "mm_only": lambda t, d: np.ones_like(t),
        "fixed_0.5": lambda t, d: np.full_like(t, 0.5),
        "gated_trace": lambda t, d: gate_trace.weights(t),
        "gated_disp": lambda t, d: gate_disp.weights(d),
        "gated_adaptive": lambda t, d: gate_adaptive.weights(d),
    }
    rank_sets, test_trace, test_disp = _evaluate(seqs, test_fns, args.seed + 100)
    weights = {
        "trace": gate_trace.weights(test_trace),
        "disp": gate_disp.weights(test_disp),
        "adaptive": gate_adaptive.weights(test_disp),
    }
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
        "adaptive_vs_fixed": paired_stats(rank_sets["fixed_0.5"], rank_sets["gated_adaptive"]),
        "adaptive_vs_id": paired_stats(rank_sets["id_only"], rank_sets["gated_adaptive"]),
        "mm_vs_id": paired_stats(rank_sets["id_only"], rank_sets["mm_only"]),
        "tracegate_vs_fixed": paired_stats(rank_sets["fixed_0.5"], rank_sets["gated_trace"]),
    }
    for key, stat in significance.items():
        print(
            f"[sig] {key:20s} ΔHR@10={stat['delta']:+.4f} "
            f"95%CI=[{stat['ci95'][0]:+.4f},{stat['ci95'][1]:+.4f}] "
            f"wins/losses={stat['wins']}/{stat['losses']} p={stat['p_value']:.3g}"
        )

    # ---- 分桶诊断
    buckets = bucket_report("trace", test_trace, rank_sets, weights, "gated_trace")
    buckets += bucket_report("disp", test_disp, rank_sets, weights, "gated_disp")
    buckets += bucket_report("adaptive", test_disp, rank_sets, weights, "gated_adaptive")
    for row in buckets:
        gate = f"{row['mean_gate']:.3f}" if row["mean_gate"] is not None else "  n/a"
        print(
            f"[bucket {row['uncertainty']:8s} {row['bucket']}] n={row['n']:6d} "
            f"unc={row['mean_unc']:.3f} g={gate} "
            f"HR@10 id={row['hr10_id']:.4f} mm={row['hr10_mm']:.4f} "
            f"fixed={row['hr10_fixed']:.4f} gated={row['hr10_gated']:.4f} "
            f"Δfixed={row['delta']:+.4f} Δid={row['delta_vs_id']:+.4f}"
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

    # ---- 前提检验：「越迷茫越该看内容」到底成不成立
    hit_id = (rank_sets["id_only"] <= 10).astype(np.float64)
    hit_mm = (rank_sets["mm_only"] <= 10).astype(np.float64)
    hit_ad = (rank_sets["gated_adaptive"] <= 10).astype(np.float64)
    premise: dict = {
        "corr_disp_vs_mm_minus_id": float(np.corrcoef(test_disp, hit_mm - hit_id)[0, 1]),
        "corr_disp_vs_adaptive_minus_id": float(
            np.corrcoef(test_disp, hit_ad - hit_id)[0, 1]
        ),
    }
    edges = np.quantile(test_disp, [0.2, 0.4, 0.6, 0.8])
    quint = np.digitize(test_disp, edges)
    for q in range(5):
        mask = quint == q
        if not mask.any():
            continue
        premise[f"q{q + 1}"] = {
            "n": int(mask.sum()),
            "mean_disp": float(test_disp[mask].mean()),
            "hr10_id": float(hit_id[mask].mean()),
            "hr10_mm": float(hit_mm[mask].mean()),
            "hr10_adaptive": float(hit_ad[mask].mean()),
            "delta_mm": float(hit_mm[mask].mean() - hit_id[mask].mean()),
            "delta_adaptive": float(hit_ad[mask].mean() - hit_id[mask].mean()),
        }
    print(
        f"[premise] corr(disp, mm-id)={premise['corr_disp_vs_mm_minus_id']:+.4f} "
        f"corr(disp, adaptive-id)={premise['corr_disp_vs_adaptive_minus_id']:+.4f}"
    )
    for q in range(1, 6):
        row = premise.get(f"q{q}")
        if row:
            print(
                f"[premise q{q}] n={row['n']:5d} disp={row['mean_disp']:.3f} "
                f"HR@10 id={row['hr10_id']:.4f} mm={row['hr10_mm']:.4f} "
                f"adaptive={row['hr10_adaptive']:.4f} "
                f"Δmm={row['delta_mm']:+.4f} Δad={row['delta_adaptive']:+.4f}"
            )

    payload = {
        "config": vars(args),
        "gates": {
            "trace": gate_trace.as_dict(),
            "disp": gate_disp.as_dict(),
            "adaptive": gate_adaptive.as_dict(),
            "adaptive_grid": {
                "floors": list(floor_grid),
                "slopes": list(slope_grid),
                "best": best_name,
                "best_valid_hr10": best_hr,
            },
        },
        "results": results,
        "buckets": buckets,
        "significance": significance,
        "diagnostics": diag,
        "premise": premise,
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
