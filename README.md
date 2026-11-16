# edurec-engine

教育资源推荐模型引擎（独立仓库），作为 edurec-platform 的离线模型服务。

**交付形态：离线训练 + 批量推理。** engine 消费平台导出的数据快照，产出推荐结果文件，
由 platform 导入其 `recommendations` 缓存表。两仓库之间只有文件交接，不共享数据库与代码；
唯一的可选在线能力是知识生成 HTTP 服务（见下文），不参与推荐主链路。

**为什么保持独立仓库**：engine 是可独立引用/复现的模型工件（含 `CITATION.cff`），
platform 是 Go + Vue 应用；双仓让 PyTorch/数据集与 Web 工具链互不污染，边界由
[`contracts/v1`](contracts/v1/README.md) 固化。若未来「一次功能改动需同时改两仓」
的频率达到每周 ≥1 次，再评估 monorepo（决策见 platform 的 `docs/design/model-design.md` #37）。

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

设计文档见 [`docs/design/model-design.md`](docs/design/model-design.md)，当前数据流见 [`docs/contracts/data-flow.md`](docs/contracts/data-flow.md)。
核心实测结论（MOOCCube，5000 活跃用户 / 706 门课，1 vs 99 采样协议）：

| 方法 | HitRate@5 | HitRate@10 | NDCG@10 | MRR |
|---|---|---|---|---|
| 训练后语义双塔 | 0.2864 | 0.4432 | 0.1710 | 0.1947 |
| 纯文本相似度（不训练） | 0.2648 | 0.3866 | 0.1604 | 0.1972 |
| 纯热门排序 | 0.3586 | 0.4930 | 0.2174 | 0.2509 |
| **语义 + 热度融合（默认配置）** | **0.4108** | **0.5758** | **0.2534** | **0.2791** |
| 随机 | 0.0495 | 0.0990 | – | – |

**两点必须知道**：

1. **纯语义单独用弱于纯热门**；融合后才超过热门。权重是本数据集实测定的
   （`0.6 / 0.0 / 0.4`），**不是**论文的 `0.7 / 0.2 / 0.1`。
2. **概率式（高斯分布）扩展已实现但不启用**——实测无收益，根因与后续方向见
   [`docs/design/gaussian-recall-status.md`](docs/design/gaussian-recall-status.md)。

## 当前阶段（2026-11）：序列主干 + 多模态 + 知识图谱

上面那套语义双塔的结论说明「纯文本语义」不适合当主路（纯语义弱于纯热门）。
现阶段的路线换成三段式：

```
                 ┌── 知识图谱：EMBEDD-ER 移植（src/kg/）→ 300 维课程/概念嵌入
STOSA（序列主干） ─┤
   物品 = N(μ,Σ)  └── 多模态：PRISM 复现（封面/关键帧 → CLIP 512d）
                          │
                          └─ 融合线已证伪并撤回（见父目录 失败记录-图文融合-2026-11-16.md）
```

| 模块 | 状态 | 证据 |
| --- | --- | --- |
| STOSA（主干） | Amazon Beauty 复现到论文 ~90%（MRR 0.0325 vs 0.0360），优于 SASRec +11.8% | `STOSA/REPRODUCTION_REPORT.md` |
| PRISM（多模态） | 三 seed 复核：R@10 +5.6%（t=3.92）、N@10 +13.6%（t=7.09） | `PRISM/RUN_LOCAL.md` |
| EMBEDD-ER（知识图谱） | 425 个概念 / 1027 条先修边 → 300 维嵌入 | `src/kg/`、`docs/handoff-2026-11-09.md` |
| 融合（已证伪） | 四轮实验、六个证伪：1v99 下自适应门控看着有效，全量排序下 HR@10 −0.0708（p=1.7e−143），冷启动 −0.0547 | 父目录 `失败记录-图文融合-2026-11-16.md` |

**复现工作区**：`STOSA/`、`PRISM/` 只提交代码与文档，数据集、CLIP 特征、checkpoint 与论文 PDF
走 `.gitignore`，重新生成的命令见 `STOSA/UPSTREAM.md`、`PRISM/RUN_LOCAL.md`。
两者的图文融合尝试（2026-11）已证伪并撤回，结论与数字见父目录
`失败记录-图文融合-2026-11-16.md`；引擎侧不再保留融合脚本。

## 环境与测试

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .[dev]          # 推荐引擎 + 测试
pip install -e .[dev,kg]       # 另装知识生成服务的最小依赖（rdflib/sklearn）
python -m pytest
```

## 数据源

| 数据源 | 命令 | 说明 |
|---|---|---|
| `mooccube` | 数据放在 `dataset/MOOCCube/` | 真实教育域数据，**推荐用它验证模型**；数据集需自行下载（不入库） |
| `sim` | `python -m scripts.gen_sim_data` | 模拟数据，打通流水线与回归测试 |
| `platform` | 由 platform 的 `export_snapshot` 产出，拷入 `dataset/platform_snapshot/<run_id>/` | 平台真实快照，见 [`docs/contracts/platform-contract.md`](docs/contracts/platform-contract.md) |

> `dataset/`、`model/` 都不入库。`mooccube` 数据缺失时命令会明确报错，不会静默降级。

## 训练

```bash
python -m scripts.train_semantic --data-source mooccube
python -m scripts.train_semantic --data-source mooccube --sparse-tail 2   # 加跑历史稀疏场景
```

产出：

- `model/semantic_recall.pt`：模型 + 编码器版本 + 语料指纹 + `item_ids`
- `model/metrics_semantic.json`：本次指标（含纯文本 / 热门两条对照基线）

`--data-source` **必填**，避免拿错轨道训练。

## 批量推理

```bash
python -m scripts.run_batch_infer \
  --data-source mooccube \
  --out-dir model
```

产出：

| 文件 | 内容 |
|---|---|
| `model/recommendations.json` | 主文件：`{平台 user_id: [平台 resource_id, …]}`，按相关性降序、已去重、覆盖全量用户 |
| `model/recommendations.meta.json` | 旁挂信封：分数、模型标识、`run_id`、快照 `run_id`、质量权重 |

推理前会校验编码器版本与**资源语料指纹**；写出后立即跑契约校验器，违规即非零退出。
`--data-source` 同样必填；产物要给 platform 时必须用 `platform` 轨道。

## 交接契约（可执行）

规范文档：[`docs/contracts/platform-contract.md`](docs/contracts/platform-contract.md)
可执行版本：[`contracts/v1/`](contracts/v1/README.md)（JSON Schema + golden fixtures + 校验器）

```bash
# 校验 platform 导出的快照（meta.json + 5 个 CSV + sha256）
python contracts/v1/validate.py snapshot dataset/platform_snapshot/<run_id>

# 校验推荐产物（主文件 + 可选信封）
python contracts/v1/validate.py recommendations \
  model/recommendations.json --envelope model/recommendations.meta.json

# 一次校验两侧
python contracts/v1/validate.py all \
  --snapshot dataset/platform_snapshot/<run_id> \
  --recommendations model/recommendations.json
```

平台仓库通过 `scripts/check-contract-sync.sh` 复用同一份 fixtures，避免两侧各写各的。

## 与 platform 配合（日常，无需手动）

engine 是 platform 的离线推荐引擎。日常使用在 platform 仓库执行：

```bash
cd ../edurec-platform
bash scripts/handoff.sh               # 导出快照 → 校验 → 训练 → 推理 → 校验产物 → 回传导入
bash scripts/handoff.sh --infer-only  # 复用已有模型，只导出 → 推理 → 导入
```

`handoff.sh` 会自动用本仓库的 `.venv` 完成训练与全量推理，并把
`model/recommendations.json` 拷回 platform 导入。

## 独立演示

```bash
python -m scripts.gen_sim_data        # 模拟数据 → dataset/sim
python -m scripts.train_semantic --data-source sim --model-dir model --epochs 3
python -m scripts.run_batch_infer --data-source sim --model-dir model --out-dir model
```

## 知识生成服务（可选在线能力）

`scripts/kg_server.py` 把 `src/kg/generate.py` 的「标题/描述 → 概念」能力封装成 HTTP 服务。
平台在搜索或知识点详情里可实时请求，拿到结果后落库展示。生成算法、Wikifier 调用与
模型依赖都留在 engine，平台不执行 Python。

```bash
# 仓库根目录；依赖与 model/ 需先准备好（model/ 不入库）
.venv/bin/python scripts/kg_server.py --host 127.0.0.1 --port 8090
```

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 服务状态 + `contract_version` |
| POST | `/knowledge/generate` | `{title, description, point_code}` → `{concepts:[{name, page_rank}]}` |
| POST | `/knowledge/generate_batch` | 批量生成，单条失败不阻断整批 |

契约与版本规则见 [`docs/contracts/kg-service-contract.md`](docs/contracts/kg-service-contract.md)。
平台侧配置 `knowledge.generator_url`（默认 `http://127.0.0.1:8090`）；engine 未启动时
只影响「自动生成概念」，不影响推荐、搜索等其他功能。

## 目录结构

```
contracts/v1/          交接契约：schema、fixtures、校验器
src/engine/            推荐引擎：data → features → models → pipeline
src/kg/                知识图谱/概念生成（生产路径；research/ 为研究脚本，不在运行时）
scripts/               训练/推理/服务入口
docs/                  设计、契约、数据流、研究状态
tests/                 engine + 契约测试
```

- 参数集中在 `src/engine/config.py`，`seed` 固定可复现。
- 文本编码器可插拔：本地降级实现（无网络依赖）↔ 真 transformer
  （`encoder_kind="sentence_transformer"`，需先装 `sentence-transformers` 并缓存模型）。
- 引用与来源见 [`docs/reference/references.md`](docs/reference/references.md) 与 [`CITATION.cff`](CITATION.cff)；本仓库不再保存论文原文抽取文本。

## 文档

- [docs/README.md](docs/README.md) —— **文档地图（从这里开始）**
- [docs/architecture.md](docs/architecture.md) —— engine 双角色与工程边界
- [docs/backlog.md](docs/backlog.md) —— engine 待实现清单与建议顺序
- [docs/contracts/README.md](docs/contracts/README.md) —— 跨仓契约总览（C1 / C2）
- [docs/design/model-design.md](docs/design/model-design.md) —— 推荐模型设计

详细设计与参考在 `docs/design/`、`docs/reference/`；契约权威在 `docs/contracts/`。
