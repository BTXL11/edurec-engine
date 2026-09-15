# edurec-engine

教育资源推荐模型引擎（独立仓库）：双塔 DSSM 召回 → 多任务 DeepFM 精排 → MMR 重排。

## 环境与测试

```bash
pip install -e .[dev]
python -m pytest tests
```

## 模拟数据演示

```bash
python -m scripts.gen_sim_data
python -m scripts.train_all --data-source sim
python -m scripts.run_batch_infer     # → model/recommendations.json
```

## 平台真实数据

```bash
# 快照由 platform export_snapshot 产出后手动拷入（见 platform docs/data-handoff.md）
ENGINE_SNAPSHOT_DIR=dataset/platform_snapshot/<run_id> python -m scripts.train_all --data-source platform
ENGINE_SNAPSHOT_DIR=dataset/platform_snapshot/<run_id> python -m scripts.run_batch_infer
```

## 说明

- 产物 `recommendations.json`：`{原始用户ID: [原始资源ID,…]}`，覆盖全量用户，冷启动走热门。
- 训练与推理须用**同一份快照**（同一 `<run_id>`）。
- 数据源：`sim` / `movielens` / `platform`；MovieLens 先跑 `python -m scripts.load_movielens`。
- 参数见 `src/engine/config.py`（默认 top_n=20，seed 可复现）。
- `dataset/`、`model/` 不入库；engine 只读写本仓库，与 platform 交接一律手动拷贝。
- 与 platform 的交接契约（快照与推荐结果的文件格式、换模型时的准则）见 `docs/platform-contract.md`。
- 架构详见 `docs/design.md`。
