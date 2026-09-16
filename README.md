# edurec-engine

教育资源推荐模型引擎（独立仓库），作为 edurec-platform 的离线模型服务。

> **当前状态：重建中（`feat/rebuild-model` 分支）**
> 上一版「双塔 DSSM 召回 + 多任务 DeepFM 精排 + MMR 重排」的模型、特征、流水线代码
> 已整体移除，仓库只保留**骨架与数据契约**，用于构建全新模型。旧版实现见 git 历史
> （截至 `9563a65`）。

## 保留了什么

```
src/engine/
  config.py            # 统一配置（dataclass + yaml）
  data/
    schema.py          # 统一数据 schema：User / Resource / Behavior / Rating / DataBundle
    io.py              # 模拟数据 CSV 落盘与读取
    simulator.py       # 行为模拟器（注入隐藏结构，可复现）
    movielens.py       # MovieLens-1M 加载器 → 统一 schema
    platform.py        # platform 数据快照加载器（契约校验）
scripts/
  gen_sim_data.py      # 生成模拟数据 → dataset/sim
  load_movielens.py    # 下载并转换 MovieLens-1M → dataset/ml_processed
```

三路数据源（模拟器 / MovieLens / platform 快照）都在入口归一为 `DataBundle`，
新模型的下游只需依赖这套 schema。

## 已移除

- `src/engine/features/`、`src/engine/models/`、`src/engine/pipeline/` 及对应单测
- `src/engine/data/preprocess.py` 中与旧流水线绑定的部分（清洗 / 词表 / 切分 / 样本构建）
- `scripts/train_all.py`、`scripts/run_batch_infer.py`

## 环境与测试

```bash
pip install -e .[dev]
python -m pytest tests
```

## 数据准备

```bash
python -m scripts.gen_sim_data        # 模拟数据 → dataset/sim
python -m scripts.load_movielens      # 公开集 → dataset/ml_processed（首次会联网下载）
```

## platform 真实数据

快照由 platform 的 `export_snapshot` 产出后手动拷入 `dataset/platform_snapshot/<run_id>/`，
契约与交接流程见 `docs/platform-contract.md`。

## 说明

- 数据源：`sim` / `movielens` / `platform`，三路统一为 `DataBundle`。
- `dataset/`、`model/` 不入库；engine 只读写本仓库，与 platform 交接一律手动拷贝。
- 参数见 `src/engine/config.py`（seed 可复现）。
- 架构详见 `docs/design.md`（旧版设计，重建新模型时同步修订）。
