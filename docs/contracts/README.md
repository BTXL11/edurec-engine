# 契约总览

engine 与 platform 之间只有两条契约。任何跨仓交互都必须落在其中一条里；
新增第三条（例如在线推荐）必须先写 ADR 与契约文档。

| 契约 | 方向 | 传输 | 权威文档 | 可执行版本 |
|---|---|---|---|---|
| **C1 离线工件** | platform ⇄ engine | 快照 CSV / 推荐 JSON（文件或对象存储） | [`platform-contract.md`](platform-contract.md) | [`../../contracts/v1/`](../../contracts/v1/README.md) |
| **C2 在线 KG** | platform → engine | HTTP/JSON（仅内网） | [`kg-service-contract.md`](kg-service-contract.md) | [`../../contracts/v1/kg-service.schema.json`](../../contracts/v1/kg-service.schema.json) |

## C1：离线工件（主链路）

```
platform 导出快照 → engine 训练/推理 → 推荐结果 → platform 导入缓存
```

- **与模型无关**：platform 只认「平台 ID 进、平台 ID 出」；
- 快照带 `contract_version=1` + `files_sha256`；
- 推荐主文件只能是 `{user_id: [resource_id, …]}`，元信息放旁挂信封；
- 出口必须跑 `contracts/v1/validate.py`；校验失败拒绝交付；
- 当前数据流与失效语义见 [`data-flow.md`](data-flow.md)。

## C2：在线 KG（可选旁路）

- 无状态 HTTP 服务，只服务“标题/描述 → 概念”；
- 生产必须启用 Bearer token、请求体/批量/并发上限、liveness/readiness；
- platform 侧有超时、并发上限与熔断，**不自动重试生成**；
- KG 不可用只影响自动生成概念，不影响推荐、搜索、学习路径。

## 版本与变更流程

1. `contract_version` 当前为 `1`；
2. **加法兼容**（新增可选字段、质量提升）不升版本；
3. **破坏性变更**（删字段、改类型、改路径/错误语义）先升版本，再同步两侧；
4. 改契约的顺序：`contracts/v1` schema/fixtures → 本文档 → platform 副本
   （`scripts/check-contract-sync.sh`）→ `verify-all.sh`。

## 不在这里的交互

- 在线推荐（C3）：明确暂不引入；若引入，作为新的 `engine-serving` 部署单元与独立契约。
- engine 不访问 platform 数据库/HTTP；platform 不直接执行 engine 代码。
