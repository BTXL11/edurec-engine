# 文档地图

> 先读核心文档，细节按目录查阅。文档与代码冲突时，以 `contracts/` 下的契约为准。

## 核心（先读这三份）

| 文档 | 回答什么问题 |
|---|---|
| [`architecture.md`](architecture.md) | engine 是什么：一个仓库、两个部署单元（离线 batch / 在线 KG），边界与演进路线 |
| [`backlog.md`](backlog.md) | engine 待实现清单、优先级与建议顺序 |
| [`contracts/README.md`](contracts/README.md) | 与 platform 怎么交互：C1 离线工件、C2 在线 KG |
| [`design/model-design.md`](design/model-design.md) | 推荐模型怎么工作：语义双塔、质量融合、评估协议 |

## 契约（跨仓接口，权威）

| 文档 | 说明 |
|---|---|
| [`contracts/platform-contract.md`](contracts/platform-contract.md) | **C1 权威**：快照进、推荐出；与模型无关 |
| [`contracts/kg-service-contract.md`](contracts/kg-service-contract.md) | **C2 权威**：知识生成 HTTP 服务（可选在线能力） |
| [`contracts/data-flow.md`](contracts/data-flow.md) | 当前实现的数据流、命令、失效与降级 |
| [`../contracts/v1/README.md`](../contracts/v1/README.md) | 契约的可执行版本：schema + fixtures + 校验器 |

## 设计细节

| 文档 | 说明 |
|---|---|
| [`design/model-design.md`](design/model-design.md) | 模型设计、数据层、编码器、训练与评估 |
| [`design/gaussian-recall-status.md`](design/gaussian-recall-status.md) | 概率式表示的负面结论与后续方向 |
| [`design/architecture-review.md`](design/architecture-review.md) | 分层/数据访问/缓存/配置的评估与重构记录 |
| [`../src/kg/README.md`](../src/kg/README.md) | 知识图谱/概念生成的生产路径 |
| [`../src/kg/research/README.md`](../src/kg/research/README.md) | 研究脚本（不在运行时路径） |

## 参考

| 文档 | 说明 |
|---|---|
| [`reference/references.md`](reference/references.md) | 引用来源（CourseHub / EMBEDD-ER / 数据集） |
| [`../CITATION.cff`](../CITATION.cff) | 软件引用信息 |

## 推荐阅读路径

- **第一次接触**：根 [`README.md`](../README.md) → 本文 → `architecture.md` → `contracts/README.md`
- **改模型/训练**：`design/model-design.md` → `contracts/platform-contract.md` → `design/architecture-review.md`
- **接入 platform / 改契约**：`contracts/platform-contract.md` → `contracts/data-flow.md` → `../contracts/v1/`
- **部署/运维 KG 服务**：`architecture.md` §3 → `contracts/kg-service-contract.md`

## 文档维护规则

1. 核心文档只保留“必须知道”的结论，细节下沉到 `design/` 或 `guides/`；
2. 契约变更先改 `contracts/v1`，再改 `contracts/*.md`，最后同步 platform；
3. 新增文档必须在上面的地图里登记，否则视为不存在。
