# 目标架构：engine 的双角色与工程边界

> 状态：设计基线（2026-11-12）
> 跨仓系统视图：[`edurec-platform/docs/architecture.md`](https://github.com/Shionyori/edurec-platform/blob/main/docs/architecture.md)
> 约束清单：[`edurec-platform/docs/constraints.md`](https://github.com/Shionyori/edurec-platform/blob/main/docs/constraints.md)
> 主契约：[`platform-contract.md`](contracts/platform-contract.md)；在线契约：[`kg-service-contract.md`](contracts/kg-service-contract.md)

---

## 0. 结论

engine 不再是「纯离线脚本仓库」，但也不应该变成一个混合部署的胖服务。目标形态是：

> **一个仓库，两个可部署单元，共享模型资产与代码，分离进程、镜像、资源与失败域。**

| 单元 | 入口 | 运行形态 | 资源画像 | 失败影响 |
|---|---|---|---|---|
| `engine-batch` | `python -m scripts.train_semantic` / `run_batch_infer` | Job，跑完即退出 | CPU/GPU 高、内存大、无入站 | 本轮推荐不更新，旧结果继续服务 |
| `engine-kg` | `python scripts/kg_server.py` | 常驻 HTTP 服务 | CPU/IO 中、无状态、可多副本 | 只影响自动生成概念 |

离线推荐是**主链路**；KG 是**旁路能力**。两者不互相 import、不共享运行时状态。

### 0.1 命名约定：域包 ≠ 部署单元

当前命名容易混淆，统一按三层理解：

| 层 | 命名 | 当前 | 说明 |
|---|---|---|---|
| **域包** | 名词 | `src/engine`（推荐域）、`src/kg`（知识域） | 代码归属；两者是对等的 Python 包 |
| **入口** | 动词/场景 | `scripts/train_semantic.py`、`run_batch_infer.py`、`kg_server.py` | 可执行入口 |
| **部署单元** | 运行时 | `engine-batch`（Job）、`engine-kg`（Service） | 进程/镜像/资源/失败域 |

**结论：`src/engine` 不是 `engine-batch`。** `src/engine` 是推荐域的包；
`engine-batch` 是“入口脚本 + `src/engine` + 模型资产”组成的离线 Job。
同理，`src/kg` 是知识域包，`engine-kg` 是它在线上运行时的部署单元。

命名目标（后续可独立执行，不影响当前结构）：

| 概念 | 当前 | 建议的最终名 |
|---|---|---|
| 推荐域包 | `src/engine/` | `src/recommendation/`（`engine` 保留一个版本的兼容别名） |
| 知识域包 | `src/kg/` | `src/knowledge/`（`kg` 保留别名） |
| 离线部署 | `engine-batch` | 不变（`batch` 是运行模式，不是域） |
| 在线部署 | `engine-kg` | 不变（或 `engine-knowledge`） |

本轮已完成的低风险部分：`src/kg` 已规范为真正的 `kg` 包（`kg.*` 绝对导入、
去掉 `sys.path` 注入），与 `src/engine` 的包结构对等；目录重命名列入
[`backlog.md`](backlog.md) 的 `ENG-16`。

---

## 1. 组件视图

```
┌──────────────────────── engine repo ────────────────────────┐
│                                                             │
│  src/engine/           离线推荐域（无网络）                  │
│    data → features → models → pipeline                      │
│        ▲                                                    │
│        │  scripts/train_semantic.py                         │
│        │  scripts/run_batch_infer.py                        │
│        │                                                    │
│  src/kg/               知识图谱/概念域                       │
│    server.py（在线入口）                                     │
│    generate.py, recommender/pipeline.py（生产路径）          │
│    research/（一次性脚本，不在运行时）                        │
│                                                             │
│  contracts/v1/         交接契约（唯一跨仓接口）               │
│  model/                模型/权重/缓存（不入库）               │
└─────────────────────────────────────────────────────────────┘
```

依赖规则：

- `src/engine` 只依赖 `src/engine` 内部模块与标准科学计算栈；
- `src/kg` 的生产路径不依赖 `src/engine`；
- 两者唯一的交汇点是 platform 的消费侧（离线工件 / 在线 HTTP），不是代码调用。

---

## 2. 离线 batch 单元

### 2.1 运行契约

```
输入：快照目录/URI（contract_version=1，带 files_sha256）
  → 训练（可选）→ 全量推理
  → recommendations.json + recommendations.meta.json
  → 出口契约校验
输出：不可变工件；校验失败拒绝交付
```

### 2.2 工程约束

| ID | 约束 | 检查 |
|---|---|---|
| E-B1 | **MUST** 离线入口不 import `http.server` / `src/kg`；可在无网络环境完成训练与推理 | `tests/test_architecture.py` |
| E-B2 | **MUST** 训练与推理使用同一快照与同一编码器口径；语料指纹不一致即失败 | checkpoint 校验 + `run_batch_infer` |
| E-B3 | **MUST** 输出先写 `.tmp` 再原子重命名；不得交付半份文件 | `write_handoff` + 测试 |
| E-B4 | **MUST** 出口跑 `contracts/v1` 校验；违规非零退出 | `run_batch_infer` + `handoff.sh` |
| E-B5 | **MUST** 覆盖全量用户、自行去重、按相关性降序、使用平台原始 ID | 契约测试 |
| E-B6 | **MUST** Job 可重复执行、幂等；失败不产生可导入的半成品 | 契约 + 调度器重试语义 |
| E-B7 | **SHOULD** 输出结构化运行摘要（run_id、快照、行数、耗时、校验结果），供调度器/告警采集 | 运行日志评审（P2） |

### 2.3 部署画像

- 无入站端口、无 Service/Ingress；
- 由 CronJob / Argo Workflow / CI Job / 手动触发；
- 模型与数据目录只读挂载；输出目录可写；
- 资源限制按训练/推理分别配置（训练可 GPU，推理 CPU 即可）。

---

## 3. 在线 KG 单元

### 3.1 运行契约

```
GET  /health   liveness
GET  /ready    readiness（可选 ready_file）
POST /knowledge/generate
POST /knowledge/generate_batch
```

无状态、不落库、不写 platform；请求结束即删除临时文件。

### 3.2 生产加固（已实现）

| 能力 | 实现 |
|---|---|
| 鉴权 | `--token` / `KG_SERVICE_TOKEN`；非空时要求 `Authorization: Bearer` |
| 请求体上限 | `--max-body-bytes`，默认 1 MiB；超限 413 |
| 批量上限 | `--max-batch-items`，默认 50；超限 400 |
| 并发上限 | `--max-concurrency`，默认 4；饱和返回 429 |
| 请求关联 | 透传/生成 `X-Request-ID`，响应回写；日志带 request_id |
| 契约版本 | 响应头 `X-Contract-Version: 1` |
| 健康/就绪 | `/health` 进程存活；`/ready` 检查 `--ready-file` |
| 日志 | 结构化 JSON（路径、耗时、request_id） |
| 失败语义 | 单条批量失败不影响整批；4xx/5xx 明确区分 |

### 3.3 工程约束

| ID | 约束 | 检查 |
|---|---|---|
| E-K1 | **MUST** `scripts/kg_server.py` 是唯一网络入口；离线脚本不得启动 HTTP | `test_architecture.py` |
| E-K2 | **MUST** KG 生产路径不 import `src/engine` | `test_architecture.py` |
| E-K3 | **MUST** 生产启用 token 且仅内网暴露；非回环地址无 token 时启动告警 | `server.py` + 部署评审 |
| E-K4 | **MUST** 请求体/批量/并发有硬上限；不得无界起线程 | `ServiceConfig` + `test_kg_server.py` |
| E-K5 | **MUST** 提供 liveness/readiness，readiness 检查模型就绪标记 | `test_kg_server.py` |
| E-K6 | **MUST** 无状态：不写 platform、不持久化请求数据；临时目录请求结束即删 | 代码评审 |
| E-K7 | **MUST** 未知/缺失 `contract_version` 的响应不会被 platform 接受 | platform 客户端测试 |
| E-K8 | **SHOULD** 为 KG 服务设置独立的 CPU/内存配额与副本数；与 batch 不共享资源池 | 部署评审 |

### 3.4 部署画像

- Kubernetes Deployment 或等价常驻服务；
- 仅内网 Service，不挂公网 Ingress；
- readiness 指向模型/权重就绪标记（如 `model/.ready`）；
- 可水平扩展；无跨副本状态，扩缩容安全。

---

## 4. 数据与工件

| 数据 | 位置 | 生命周期 |
|---|---|---|
| 模型权重 / 编码器缓存 | `model/`（不入库） | 跨轮次复用；随版本管理 |
| 快照 | 输入 URI / `dataset/platform_snapshot/<run_id>/` | 只读；按保留策略清理 |
| 推荐工件 | 输出 URI / `model/recommendations.json` | 不可变；导入后可归档 |
| KG 临时文件 | 系统临时目录 | 请求级，结束即删 |

**约束**：不得把 platform 数据、模型权重或推荐工件写回对方目录；交接只通过契约工件。

---

## 5. 版本与兼容

| 版本 | 位置 | 规则 |
|---|---|---|
| `contract_version` | 快照 `meta.json`、推荐信封、KG 响应 | 破坏性变更才升；读取方拒绝未知版本 |
| 包版本 | `pyproject.toml` | 代码/部署版本；与契约版本独立 |
| 模型版本 | 信封 `model.version`、checkpoint | 算法/权重；换模型不改主文件形状 |
| 编码器版本 | checkpoint `encoder_version` + `corpus_fingerprint` | 训练/推理口径一致性 |

---

## 6. 演进路线

| 阶段 | 目标 | 验收 |
|---|---|---|
| P1（本轮） | KG 服务生产加固 + 双角色边界测试 + 文档 | `pytest` 全绿；`verify-all.sh` 全绿 |
| P2 | 拆分构建产物：`engine-batch` 与 `engine-kg` 两个镜像/入口；模型资产只读挂载 | 两个镜像可独立启动；batch 无入站端口 |
| P3 | 接入工件存储（S3/MinIO）客户端与 manifest；批次可移植 | 不依赖相邻目录完成一轮刷新 |
| P4 | 模型注册表/版本化资产（可选）；在线推荐 `engine-serving` 单独评审 | 独立 ADR 与压测 |
