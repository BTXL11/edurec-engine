"""批量推理入口：产出对齐 platform 契约的推荐结果。

用法：
    python -m scripts.run_batch_infer --data-source mooccube
    python -m scripts.run_batch_infer --data-source platform --snapshot-dir <dir>

产出（`docs/platform-contract.md` 方向 B）：
    model/recommendations.json        主文件，{平台 user_id: [平台 resource_id, …]}
    model/recommendations.meta.json   旁挂信封（分数、模型标识、快照 run_id），可选
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
from engine.features.encoder import build_encoder, resource_text
from engine.models.recall.semantic import TwoTowerSemantic
from engine.pipeline.infer_batch import infer_batch, write_handoff

RECALL_CKPT = "semantic_recall.pt"


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


def load_model(cfg: EngineConfig, bundle, checkpoint_path: str):
    """从 checkpoint 恢复编码器与双塔。

    编码器是统计型的（IDF 来自语料），因此必须用**同一份资源语料**重新拟合；
    再用 checkpoint 里的 `encoder_version` 校验一致性——版本里含语料指纹，
    换了语料或改了 n-gram 参数都会不匹配，此时应当明确报错而不是给出错误推荐。
    """
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(f"找不到模型文件 {checkpoint_path}，请先运行 "
                         f"`python -m scripts.train_semantic`")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    item_texts = [resource_text(r) for r in
                  sorted(bundle.resources, key=lambda x: x.resource_id)]
    encoder = build_encoder(cfg)
    encoder.fit(item_texts)

    expect = ckpt.get("encoder_version")
    if expect and encoder.version != expect:
        raise SystemExit(
            "编码器与训练时不一致，推荐结果不可信：\n"
            f"  训练时: {expect}\n"
            f"  当前  : {encoder.version}\n"
            "（encoder_* 参数变了；请用同一份配置重新训练）")

    # 编码器的 IDF 权重来自语料，因此还要确认资源数据本身没换。
    # 只比 encoder_version 是不够的：它反映的是参数，不反映语料。
    expect_corpus = ckpt.get("corpus_fingerprint")
    if expect_corpus:
        import hashlib
        current = hashlib.sha256(
            "".join(sorted(item_texts)).encode("utf-8")).hexdigest()[:16]
        if current != expect_corpus:
            raise SystemExit(
                "资源语料与训练时不一致，推荐结果不可信：\n"
                f"  训练时指纹: {expect_corpus}\n"
                f"  当前指纹  : {current}\n"
                "（换了数据集或资源内容变了；请用同一份数据重新训练）")

    model = TwoTowerSemantic(in_dim=int(ckpt["in_dim"]),
                             out_dim=int(ckpt["out_dim"]),
                             hidden=cfg.recall_hidden_dim,
                             dropout=cfg.recall_dropout,
                             temperature=cfg.recall_temperature)
    model.student_tower.load_state_dict(ckpt["student_tower"])
    model.item_tower.load_state_dict(ckpt["item_tower"])
    model.eval()
    return model, encoder, ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", default="mooccube",
                    choices=["mooccube", "sim", "platform"])
    ap.add_argument("--snapshot-dir", default="")
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--no-envelope", action="store_true",
                    help="不写旁挂信封（主文件不受影响）")
    args = ap.parse_args()

    cfg = EngineConfig(data_source=args.data_source)
    if args.snapshot_dir:
        cfg.snapshot_dir = args.snapshot_dir
    if args.top_n is not None:
        cfg.top_n = args.top_n

    t0 = time.time()
    bundle = load_data(cfg)
    print(f"[data] {cfg.data_source}: 用户 {len(bundle.users)} "
          f"资源 {len(bundle.resources)} 行为 {len(bundle.behaviors)} "
          f"({time.time() - t0:.1f}s)")

    model, encoder, ckpt = load_model(cfg, bundle,
                                      os.path.join(cfg.model_dir, RECALL_CKPT))

    # 推理用「已知的全部历史」（要预测的是未来）；不传 split，因此不会只看训练期
    item_ids = ckpt.get("item_ids") or sorted(r.resource_id for r in bundle.resources)
    t0 = time.time()
    result = infer_batch(model, encoder, bundle, cfg, item_ids=item_ids)
    print(f"[infer] {time.time() - t0:.1f}s  覆盖 {len(result.recommendations)} 用户 "
          f"（其中冷启动 {result.cold_start_users}）")

    out_path = os.path.join(cfg.model_dir, "recommendations.json")
    envelope_path = os.path.join(cfg.model_dir, "recommendations.meta.json")
    envelope = {
        "contract_version": 1,
        "generated_at": int(time.time()),
        "model": {"name": ckpt.get("model_name", "semantic_two_tower"),
                  "version": "v1"},
        "method": {"recall": "semantic_two_tower",
                   "rank": "quality_fusion",
                   "rerank": "seen_filter + top_n"},
        "encoder": encoder.version,
        "top_n": cfg.top_n,
        "users_count": len(result.recommendations),
        "cold_start_users": result.cold_start_users,
        "quality_weights": {"semantic": cfg.quality_w_semantic,
                            "rating": cfg.quality_w_rating,
                            "popularity": cfg.quality_w_popularity},
    }
    snap = cfg.snapshot_dir or (cfg.mooccube_dir if cfg.data_source == "mooccube" else "")
    if snap:
        envelope["snapshot_run_id"] = os.path.basename(os.path.normpath(snap))

    write_handoff(result, out_path,
                  None if args.no_envelope else envelope_path,
                  envelope)

    # 出口自检：契约的几条硬约束在这里当场核对，避免把坏文件交给 platform
    with open(out_path, encoding="utf-8") as f:
        raw = f.read()
    payload = json.loads(raw)
    assert not raw.startswith("\ufeff"), "主文件带了 BOM"
    assert all(k.isdigit() for k in payload), "存在非数字键"
    assert all(isinstance(v, list) and all(isinstance(x, int)
                                          for x in v) for v in payload.values()), \
        "值必须是整数数组"
    n_empty = sum(1 for v in payload.values() if not v)
    print(f"[out] {out_path}")
    if not args.no_envelope:
        print(f"[out] {envelope_path}")
    print(f"[check] 用户 {len(payload)}  空列表 {n_empty}  "
          f"平均条数 {np.mean([len(v) for v in payload.values()]):.1f}")


if __name__ == "__main__":
    main()
