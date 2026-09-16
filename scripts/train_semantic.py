"""阶段一训练与评估入口：语义双塔召回。

用法：
    python -m scripts.train_semantic --data-source mooccube
    python -m scripts.train_semantic --data-source sim --epochs 3

产出：
    model/semantic_recall.pt        双塔权重 + 编码器版本 + 配置摘要
    model/metrics_semantic.json     本次评估指标（含三条对照基线）
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from engine.config import EngineConfig
from engine.data.io import load_bundle
from engine.data.mooccube import load as load_mooccube
from engine.data.platform import load as load_platform
from engine.features.encoder import build_encoder
from engine.models.recall.trainer import build_interactions, train_two_tower
from engine.pipeline.recall_pipeline import (
    evaluate_popularity_sampled, evaluate_sampled,
    evaluate_text_similarity_sampled, split_user_interactions,
)


def load_data(cfg: EngineConfig):
    if cfg.data_source == "mooccube":
        return load_mooccube(cfg.mooccube_config())
    if cfg.data_source == "platform":
        if not cfg.snapshot_dir:
            raise SystemExit("--data-source platform 需要 --snapshot-dir 指定快照目录")
        return load_platform(cfg.snapshot_dir)
    if cfg.data_source in ("sim", "movielens"):
        return load_bundle(os.path.join(cfg.data_dir, "sim"))
    raise SystemExit(f"未知的数据源: {cfg.data_source}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", default="mooccube",
                    choices=["mooccube", "sim", "platform"])
    ap.add_argument("--snapshot-dir", default="")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--sample-mode", default=None,
                    choices=["last_prefix", "all_prefixes"])
    ap.add_argument("--negatives", type=int, default=None,
                    help="评估时每用户采样的负样本数（0 = 全量候选）")
    ap.add_argument("--no-baselines", action="store_true",
                    help="跳过纯文本相似度与热门基线（省时间）")
    args = ap.parse_args()

    cfg = EngineConfig(data_source=args.data_source)
    if args.snapshot_dir:
        cfg.snapshot_dir = args.snapshot_dir
    if args.epochs is not None:
        cfg.recall_epochs = args.epochs
    if args.sample_mode is not None:
        cfg.recall_sample_mode = args.sample_mode
    if args.negatives is not None:
        cfg.eval_n_negatives = args.negatives

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    t0 = time.time()
    bundle = load_data(cfg)
    print(f"[data] {cfg.data_source}: 用户 {len(bundle.users)} "
          f"资源 {len(bundle.resources)} 行为 {len(bundle.behaviors)} "
          f"({time.time() - t0:.1f}s)")

    interactions = build_interactions(bundle)
    split = split_user_interactions(interactions, cfg.train_ratio, cfg.val_ratio)
    train_ints = [it for acts in split.train.values() for it in acts]
    item_ids = sorted(r.resource_id for r in bundle.resources)
    print(f"[split] 训练用户 {len(split.train)} 训练交互 {len(train_ints)}")

    encoder = build_encoder(cfg)
    t0 = time.time()
    model = train_two_tower(encoder, train_ints, item_ids, cfg)
    print(f"[train] {time.time() - t0:.1f}s  "
          f"loss {model.loss_history[0]:.4f} -> {model.loss_history[-1]:.4f}")

    n_neg = cfg.eval_n_negatives
    report: dict = {
        "data_source": cfg.data_source,
        "encoder": encoder.version,
        "model": {"name": "semantic_two_tower", "version": "v1"},
        "loss_history": model.loss_history,
        "eval_n_negatives": n_neg,
        "sample_mode": cfg.recall_sample_mode,
    }

    t0 = time.time()
    report["semantic_two_tower"] = evaluate_sampled(
        model, encoder, bundle, split, item_ids, cfg, n_negatives=n_neg)
    if not args.no_baselines:
        report["baseline_text_similarity"] = evaluate_text_similarity_sampled(
            encoder, bundle, split, item_ids, cfg, n_negatives=n_neg)
        report["baseline_popularity"] = evaluate_popularity_sampled(
            split, item_ids, cfg, n_negatives=n_neg)
    print(f"[eval] {time.time() - t0:.1f}s")

    print("\n=== 指标（采样候选协议）===")
    keys = [k for k in ("hitrate@5", "hitrate@10", "hitrate@20", "ndcg@10", "mrr")
            if k in report["semantic_two_tower"]]
    header = f"{'方法':<26}" + "".join(f"{k:>13}" for k in keys)
    print(header)
    for name in ("semantic_two_tower", "baseline_text_similarity", "baseline_popularity"):
        if name in report:
            row = report[name]
            print(f"{name:<26}" + "".join(f"{row.get(k, float('nan')):>13.4f}" for k in keys))
    m = report["semantic_two_tower"]
    if "random_hitrate@10" in m:
        print(f"{'随机':<26}" + "".join(
            f"{m.get('random_' + k, float('nan')):>13.4f}" for k in keys))

    os.makedirs(cfg.model_dir, exist_ok=True)
    torch.save({
        "student_tower": model.student_tower.state_dict(),
        "item_tower": model.item_tower.state_dict(),
        "in_dim": model.in_dim,
        "out_dim": model.out_dim,
        "encoder_version": encoder.version,
        "item_ids": item_ids,
    }, os.path.join(cfg.model_dir, "semantic_recall.pt"))
    with open(os.path.join(cfg.model_dir, "metrics_semantic.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[out] {cfg.model_dir}/semantic_recall.pt, metrics_semantic.json")


if __name__ == "__main__":
    main()
