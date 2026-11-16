# 知识生成服务契约（可选在线能力，v1）

本文是 **engine 侧**对 `scripts/kg_server.py` 的规范性说明。它独立于
[`platform-contract.md`](platform-contract.md) 的「快照进、排名出」主契约：

- 主契约是**离线批量**，是推荐结果的唯一来源；
- 本契约是**可选在线能力**，只服务「标题/描述 → 概念」的自动生成；
- engine 未启动时，平台只失去「自动生成概念」，推荐、搜索、学习路径不受影响。

机器可读的请求/响应形状见
[`../contracts/v1/kg-service.schema.json`](../../contracts/v1/kg-service.schema.json)，
样本见 [`../contracts/v1/fixtures/kg-generate.response.json`](../../contracts/v1/fixtures/kg-generate.response.json)。

## 启动

```bash
# engine 仓库根目录；model/ 与依赖需先准备好（model/ 不入库）
pip install -e .[kg]          # rdflib + scikit-learn（服务最小依赖）
.venv/bin/python scripts/kg_server.py --host 127.0.0.1 --port 8090
```

平台侧配置：

```yaml
knowledge:
  generator_url: "http://127.0.0.1:8090"
  generator_timeout_seconds: 180
  generator_token: ""              # 生产用 KNOWLEDGE_GENERATOR_TOKEN 注入
  generator_max_concurrency: 4
  generator_failure_threshold: 3   # 连续 3 次失败后熔断
  generator_circuit_cooldown_seconds: 30
```

### 生产运行约束

| 项 | 约定 |
|---|---|
| 网络 | **只在内网暴露**；不挂公网 Ingress。batch 与 KG 是两个部署单元 |
| 鉴权 | 生产启用 `--token` / `KG_SERVICE_TOKEN`；平台发送 `Authorization: Bearer <token>` |
| 请求体上限 | `--max-body-bytes`，默认 **1 MiB**；超限 413 |
| 批量上限 | `--max-batch-items`，默认 **50**；超限 400 |
| 并发上限 | `--max-concurrency`，默认 **4**；饱和返回 429 |
| 关联 ID | 平台发送 `X-Request-ID`；服务回写同名响应头，并在日志中记录 |
| 契约版本 | 每个响应带 `X-Contract-Version: 1` |
| 就绪 | `/ready` 检查 `--ready-file`；生产应指向模型/权重就绪标记 |
| 客户端行为 | 超时 + 并发上限 + 连续失败熔断；**不自动重试生成**（昂贵且非确定性） |
| 日志 | 结构化 JSON：request_id、path、duration_ms；不记录完整描述文本 |

参考 SLO（可按部署规格调整）：

- 可用性：≥ 99%（失败只影响自动生成概念）；
- 单条生成延迟：p95 ≤ 5s（冷启动/首次模型加载除外）；
- 错误率：5xx ≤ 1%；
- 熔断/降级时平台侧快速失败，不占用业务请求线程。

## 接口

### `GET /health`

```json
{
  "status": "ok",
  "engine": "edurec-engine",
  "version": "1.1",
  "contract_version": 1,
  "capabilities": ["knowledge/generate", "knowledge/generate_batch"]
}
```

### `GET /ready`

```json
{"status": "ready", "contract_version": 1}
```

- 未配置 `--ready-file` 时恒为 200（进程可接请求）；
- 配置了文件但文件不存在时返回 **503**，`status=not_ready`，并带 `reason`。
- 用途：部署时摘流；与 `/health` 的 liveness 语义分开。

### `POST /knowledge/generate`

请求：

```json
{"title": "数学分析", "description": "极限、导数与积分", "point_code": "mathematical-analysis"}
```

- `title` 必填、非空；`description` / `point_code` 可选。
- `Content-Type: application/json`，请求体必须是合法 UTF-8 JSON。

成功响应（HTTP 200）：

```json
{
  "contract_version": 1,
  "point_code": "mathematical-analysis",
  "title": "数学分析",
  "concepts": [
    {"name": "积分", "page_rank": 0.045},
    {"name": "导数", "page_rank": 0.042}
  ],
  "elapsed_seconds": 1.2
}
```

`concepts` 已按 `page_rank` 降序、去重、过滤停用词与过短名称；默认最多 30 个。

### `POST /knowledge/generate_batch`

请求：

```json
{"items": [{"title": "…", "description": "…", "point_code": "…"}]}
```

成功响应（HTTP 200）：

```json
{
  "contract_version": 1,
  "results": [
    {"contract_version": 1, "point_code": "…", "title": "…", "concepts": []},
    {"point_code": "…", "title": "…", "concepts": [], "error": "ValueError: …"}
  ]
}
```

单条失败不阻断整批：失败项带 `error`、`concepts` 为空数组，HTTP 仍为 200。

## 错误语义

| HTTP | 场景 | 响应 |
|---|---|---|
| 400 | 请求体为空/非 JSON、`title` 为空、`items` 非非空数组、批量超上限 | `{"error": "…"}` |
| 401 | 启用鉴权但缺失/错误的 `Authorization` | `{"error": "unauthorized"}` |
| 404 | 未知路径 | `{"error": "not found"}` |
| 413 | 请求体超过 `max_body_bytes` | `{"error": "…"}` |
| 415 | `Content-Type` 不是 `application/json` | `{"error": "…"}` |
| 429 | 并发已达上限 | `{"error": "too many concurrent requests"}` |
| 500 | 生成内部错误（Wikifier/模型/IO） | `{"error": "…"}` |
| 503 | `/ready` 检查未通过（模型/权重未就绪） | `{"status": "not_ready", "reason": "…"}` |

平台侧把非 2xx 视为失败，只影响当前生成任务；不会回滚已有概念。

## 版本与兼容

- `contract_version` 当前为 `1`，出现在 `/health` 与每个响应顶层。
- **允许**新增可选字段（平台忽略未知字段）、调整概念数量/排序质量。
- **破坏性变更**（删除字段、改类型、改路径、改错误语义）必须提升
  `contract_version`；平台读到不认识的版本应降级为「不自动生成」，不得猜测。
- 该服务**无状态**、不落库；幂等性由平台按「学科 + 概念名」去重保证。
  同一请求重复调用不保证返回完全相同的排序（Wikifier 在线 API 可能波动）。
- 这是唯一允许的 platform → engine 在线调用。除此之外两仓库不互相调用接口，
  也不读写对方目录（主链路只走文件交接）。
