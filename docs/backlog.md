# engine 待实现清单

> 更新：2026-11-12
> 本文件只列 **engine 自身**的待办。跨仓/部署/平台侧的总清单在
> [edurec-platform/docs/backlog.md](https://github.com/Shionyori/edurec-platform/blob/main/docs/backlog.md)。
> 关联：[`architecture.md`](architecture.md)、[`contracts/README.md`](contracts/README.md)

## 状态图例

| 状态 | 含义 |
|---|---|
| **待实现** | 设计完成，只差编码/打包 |
| **待设计** | 方向已定，方案需评审 |
| **研究** | 需实验/数据验证 |

---

## 0. 本轮已闭环

- 在线 KG 服务边界：鉴权、请求体/批量/并发上限、请求 ID、readiness、结构化日志；
- 离线/在线架构边界测试；
- 契约 v1（schema + fixtures + 校验器）与数据流文档；
- 研究脚本归档到 `src/kg/research/`，生产路径与研究报告分离；
- 文档分层（core / contracts / design / reference）。

---

## 1. 工程与交付（P1/P2）

| ID | 事项 | 状态 | 验收标准 | 依赖 |
|---|---|---|---|---|
| ENG-01 | **两个可部署产物**：`engine-batch` Job 与 `engine-kg` 服务分别构建/启动；batch 无入站端口 | 待实现 | 两个镜像/入口可独立启动；KG 有 readiness；batch 可被调度器重复执行 | SYS-02 |
| ENG-02 | **工件存储客户端**：从 S3/MinIO URI 读快照、写推荐工件（或先做本地目录适配层） | 待实现 | 不依赖相邻目录；输入/输出 URI 与校验和写入运行摘要 | SYS-01 |
| ENG-03 | **batch 运行摘要**：结构化输出 run_id / 快照 / 行数 / 耗时 / 校验结果 | 待实现 | 调度器可采集；失败可定位 | 无 |
| ENG-04 | **模型资产版本化**：checkpoint、编码器缓存、语料指纹的版本目录与保留策略 | 待设计 | 可追溯某次推荐用的模型/语料；旧版本可回滚 | ENG-01 |
| ENG-05 | **配置分组**：Data / Encoder / Recall / Quality，保留 facade 兼容 | 待实现 | 现有测试与 CLI 不变；YAML 可按组书写 | 无 |
| ENG-06 | **命名与口径清理**：`LocalTfidfSvdEncoder → LocalHashIdfEncoder`（保留别名）；需求文本统一到 `BundleAccess` | 待实现 | 三处需求文本实现收敛；等价性测试通过 | 无 |
| ENG-07 | **高斯塔处置**：补齐 `history_len` 的批量推理入口，或移入 `research/` 并从生产包排除 | 待设计 | 不再存在“能训练、不能交付”的 checkpoint | 无 |
| ENG-16 | **域目录重命名**：`src/engine → src/recommendation`、`src/kg → src/knowledge`，保留旧 import 别名一个版本 | 待设计 | 目录名与域语义一致；旧入口/import 仍可用；CI 与文档同步 | 无 |

---

## 2. 推荐能力（P3，研究）

| ID | 事项 | 状态 | 验收标准 | 依赖 |
|---|---|---|---|---|
| ENG-08 | 多路召回（语义 + 热门 + 协同/内容）与显式去重 | 研究 | 单路 vs 融合指标可对比；出口去重 100% | 无 |
| ENG-09 | 多目标排序（pClick / pComplete / pSatisfy） | 研究 | 精排 AUC/融合指标优于单信号 | ENG-08 |
| ENG-10 | 重排：多样性 / 难度梯度 / 新鲜度 | 研究 | 去重 100%；多样性/新鲜度可度量 | ENG-09 |
| ENG-11 | 真 transformer 编码器接入并重调质量权重 | 研究 | `encoder_kind=sentence_transformer` 可跑；指标与权重网格重跑 | 模型/网络环境 |
| ENG-12 | 完成信号（complete）进入训练 | 待设计 | 快照新增完成事件；训练/评估纳入 | PF-07/SYS 契约 |
| ENG-13 | 在 platform 真实快照上验证（带正文、真实行为量） | 待实现 | 产出指标与基线对比；记录数据规模 | 真实快照 |
| ENG-14 | ANN（faiss）索引 | 待设计 | 资源量增长后接口不变、延迟可控 | 规模触发 |
| ENG-15 | 在线推荐 serving（C3） | 暂不做 | 需独立部署单元、契约、压测与成本评估 | 明确立项 |

---

## 3. 建议顺序

1. **ENG-01 + ENG-02**：先把“两个部署单元 + 工件存储”落地；
2. **ENG-03 + ENG-05 + ENG-06**：低风险工程清理，提升可运维性；
3. **ENG-07**：消除高斯塔的交付歧义；
4. **ENG-08 → ENG-09 → ENG-10**：研究主线，与平台工程线并行；
5. **ENG-11 / ENG-13 / ENG-14**：按数据与环境条件推进。
