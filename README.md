# edurec-engine

教育资源推荐模型引擎（独立仓库），作为 edurec-platform 的离线模型服务。

**一期交付：离线训练 + 批量推理。** 产出推荐结果文件，由 platform 导入其 `recommendations`
缓存表。两仓库之间只有文件交接，不共享数据库与代码。

## 模型

**语义嵌入召回**：把资源正文与「由学生行为历史反推的需求」映射到同一向量空间，
用距离衡量匹配度，再融合质量信号（热度/评分）排序。参考
[CourseHub](https://doi.org/10.1109/ICNWC68145.2026.11518402)。

```
资源文本 ─► Encoder ─► 资源塔 ─┐
                              ├─ 内积打分 ─► 质量融合 ─► 过滤已交互/去重 ─► top-N
学生行为历史 ─► 需求文本 ─► Encoder ─► 需求塔 ─┘                            │
                                                     冷启动 ─► 热门兜底 ─────┘
```

设计文档见 `docs/design.md`。核心实测结论：

| 方法 | HitRate@5 | HitRate@10 | NDCG@10 | MRR |
|---|---|---|---|---|
| 训练后语义双塔 | 0.2864 | 0.4432 | 0.1710 | 0.1947 |
| 纯文本相似度（不训练） | 0.2648 | 0.3866 | 0.1604 | 0.1972 |
| 纯热门排序 | 0.3586 | 0.4930 | 0.2174 | 0.2509 |
| **语义 + 热度融合（默认配置）** | **0.4108** | **0.5758** | **0.2534** | **0.2791** |
| 随机 | 0.0495 | 0.0990 | – | – |

（MOOCCube，5000 活跃用户 / 706 门课，1 vs 99 采样候选协议）

**两点必须知道**：
1. **纯语义单独用弱于纯热门**；融合后才超过热门 17%。权重是按本数据集实测定的
   （`0.6 / 0.0 / 0.4`），**不是**论文的 `0.7 / 0.2 / 0.1`。
2. **概率式（高斯分布）扩展已实现但不启用**——实测无收益，根因与后续方向见
   `docs/gaussian-recall-status.md`。

## 环境与测试

```bash
pip install -e .[dev]
python -m pytest tests
```

## 使用

### 数据源

| 数据源 | 命令 | 说明 |
|---|---|---|
| `mooccube` | 数据已在 `dataset/MOOCCube/` | 真实教育域数据，**推荐用这个验证** |
| `sim` | `python -m scripts.gen_sim_data` | 模拟数据，打通流水线与回归测试 |
| `platform` | 手动拷入 `dataset/platform_snapshot/<run_id>/` | 平台真实快照，见 `docs/platform-contract.md` |

> ⚠️ `platform` 数据源上语义模型**目前基本失效**：快照未导出资源正文
> （`resources.description`）。补导请求见 `docs/platform-description-request.md`。

### 训练

```bash
python -m scripts.train_semantic --data-source mooccube
python -m scripts.train_semantic --data-source mooccube --sparse-tail 2   # 加跑历史稀疏场景
```

产出 `model/semantic_recall.pt`（模型 + 编码器版本 + 语料指纹）与
`model/metrics_semantic.json`（含三条对照基线）。

### 批量推理

```bash
python -m scripts.run_batch_infer --data-source mooccube
```

产出：

| 文件 | 内容 |
|---|---|
| `model/recommendations.json` | 主文件：`{平台 user_id: [平台 resource_id, …]}`，按相关性降序、已去重、覆盖全量用户 |
| `model/recommendations.meta.json` | 旁挂信封：分数、模型标识、质量权重、快照 run_id（可选） |

主文件格式受契约约束（[`docs/platform-contract.md`](docs/platform-contract.md)）：
顶层不得有任何非整数数组字段、无 BOM、**先写临时文件再原子重命名**。
推理前会校验编码器版本与**资源语料指纹**，与训练时不一致则报错退出。

## 说明

- `dataset/`、`model/` 不入库；engine 只读写本仓库。
- 参数集中在 `src/engine/config.py`，`seed` 固定可复现。
- 文本编码器可插拔：本地降级实现（无网络依赖）↔ 真 transformer
  （`encoder_kind="sentence_transformer"`，需先装 `sentence-transformers` 并缓存模型）。
- 论文原文抽取文本在 `docs/papers/`（**若仓库公开建议移除**，只留引用信息）。
