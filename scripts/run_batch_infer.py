"""批量推理入口：产出对齐 platform 契约的推荐结果。

用法：
    python -m scripts.run_batch_infer --data-source mooccube
    python -m scripts.run_batch_infer --data-source platform --snapshot-dir <dir>

产出（`docs/contracts/platform-contract.md` 方向 B）：
    model/recommendations.json        主文件，{平台 user_id: [平台 resource_id, …]}
    model/recommendations.meta.json   旁挂信封（分数、模型标识、快照 run_id），可选
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    if cfg.data_source == "sim":
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

    # 批量推理当前只实现确定性双塔。高斯塔的学生塔输入多一维（历史长度），
    # 权重形状不同；如果这里硬套 TwoTowerSemantic，会在 load_state_dict 时报错，
    # 或者更糟——静默地按缺省历史长度打分。明确拒绝，避免交付错误结果。
    tower_kind = ckpt.get("tower_kind", "deterministic")
    if tower_kind != "deterministic":
        raise SystemExit(
            f"当前批量推理不支持 tower_kind={tower_kind!r} 的 checkpoint。\n"
            "高斯双塔（阶段二）已实现并评测，但默认不启用；"
            "如需推理，请先用 --tower-kind deterministic 重新训练，"
            "或为 run_batch_infer 补齐高斯塔的批量打分入口。")

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
    # 不设默认值：产物会被 platform 导入，数据源错选会造成「导入成功但推荐没变化」。
    # 也可用 ENGINE_DATA_SOURCE / ENGINE_SNAPSHOT_DIR。
    ap.add_argument("--data-source", default=None,
                    choices=["mooccube", "sim", "platform"])
    ap.add_argument("--snapshot-dir", default="")
    ap.add_argument("--model-dir", default="",
                    help="模型目录（默认 model/；相对当前工作目录）")
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--out-dir", default="",
                    help="产物目录（默认 model/；相对当前工作目录）")
    ap.add_argument("--no-envelope", action="store_true",
                    help="不写旁挂信封（主文件不受影响）")
    args = ap.parse_args()

    data_source = args.data_source or os.environ.get("ENGINE_DATA_SOURCE")
    if not data_source:
        ap.error("必须指定 --data-source（或设置 ENGINE_DATA_SOURCE）")
    snapshot_dir = args.snapshot_dir or os.environ.get("ENGINE_SNAPSHOT_DIR", "")
    if snapshot_dir:
        args.snapshot_dir = snapshot_dir

    cfg = EngineConfig(data_source=data_source)
    if args.snapshot_dir:
        cfg.snapshot_dir = args.snapshot_dir
    if args.model_dir:
        cfg.model_dir = args.model_dir
    if args.top_n is not None:
        cfg.top_n = args.top_n
    cfg.validate()

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

    out_dir = args.out_dir or cfg.model_dir
    out_path = os.path.join(out_dir, "recommendations.json")
    envelope_path = os.path.join(out_dir, "recommendations.meta.json")
    now = int(time.time())
    envelope = {
        "contract_version": 1,
        "generated_at": now,
        "run_id": time.strftime("%Y%m%d_%H%M%S", time.localtime(now)),
        "model": {"name": ckpt.get("model_name", "semantic_two_tower"),
                  "version": "v1",
                  "encoder": encoder.version},
        "method": {"recall": "semantic_two_tower",
                   "rank": "quality_fusion",
                   "rerank": "seen_filter + top_n"},
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

    # 出口自检：直接跑 contracts/v1 校验器（不用 assert，-O 下也不会消失）。
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from contracts.v1.validate import validate_recommendations
    errors = validate_recommendations(out_path)
    if errors:
        for err in errors[:20]:
            print(f"[check] 违规: {err}", file=sys.stderr)
        raise SystemExit(
            "推荐主文件不符合 contracts/v1，已拒绝交付（文件保留供排查）")

    payload = json.loads(open(out_path, encoding="utf-8").read())
    n_empty = sum(1 for v in payload.values() if not v)
    print(f"[out] {out_path}")
    if not args.no_envelope:
        print(f"[out] {envelope_path}")
    print(f"[check] 用户 {len(payload)}  空列表 {n_empty}  "
          f"平均条数 {np.mean([len(v) for v in payload.values()]):.1f}")


if __name__ == "__main__":
    main()
