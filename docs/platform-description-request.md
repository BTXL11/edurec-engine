# 变更请求：`export_snapshot` 补导 `resources.description`

> **提交方**：engine（`edurec-engine`）
> **接收方**：platform（`edurec-platform`）
> **类型**：数据快照增列（**非破坏性**，不影响现有契约版本）
> **紧急度**：阻塞 engine 在**平台真实数据**上的模型验证
> **日期**：2026-09-16

---

## 1. 请求内容

在 `cmd/export_snapshot` 的 `resources.csv` 中**增加一列 `description`**，
取平台 `resources` 表的 `description` 字段。

**这一列在平台数据库里已经存在**（见 `edurec-platform` 设计文档 §5.2
`resources(id, title, description, type, category_id, tags JSON, metadata JSON,
avg_rating, view_count, created_at)`），只是导出时没有写出。本请求**不涉及任何库表变更**。

建议列顺序（放在 `title` 之后，语义相邻）：

```
resource_id,title,description,type,category_id,tags_json,metadata_json,avg_rating,view_count,created_at
```

## 2. 为什么需要

engine 这一版的推荐模型是**语义嵌入召回**：把资源正文与学生需求编码成向量，
用距离衡量匹配度（参考 CourseHub）。它的**全部信号来自资源正文**。

当前 `resources.csv` 里唯一可用的文本是 `title`，而真实快照的标题往往是
「《XX》课程」「第一章 绪论」这类短语，不足以承载内容语义。没有正文时，
语义模型不会报错，但推荐质量会退化到接近「热门兜底」——**等于这一版模型在平台数据上白做**。

MOOCCube 轨道之所以能验证，正是因为该数据集自带课程简介（正文均值约 400 字）。

## 3. 为什么是非破坏性变更

| 关注点 | 说明 |
|---|---|
| 契约版本 | **无需变更**。`contract_version` 保持 `1`；`description` 是**可选列**，engine 缺失时降级为空串 |
| 兼容性 | engine 的 `platform.py` 已支持该列（读到就用，读不到用 `""`），旧快照仍能正常加载 |
| 反向影响 | 无。`recommendations.json`（engine → platform）格式完全不变 |
| 平台侧代码 | 只需在导出逻辑里多写一列；不需要改任何表结构、接口或缓存逻辑 |

> engine 侧实现见 `src/engine/data/platform.py`（`description` 读取）与
> `docs/platform-contract.md`「方向 A · resources.csv」。
> 单测 `tests/test_platform_loader.py` 同时覆盖「有该列」与「无该列」两种快照。

## 4. 格式要求（重要：本列**必然包含换行**）

这是唯一需要特别小心的地方。engine 实测 MOOCCube 706 条正文中：

| 特征 | 条数 |
|---|---|
| 含**换行** | **467 / 706（66%）** |
| 含逗号 | 48 |
| 含双引号 | 10 |
| 长度 | 均值 404 字、中位 343、最大 2212 |

因此：

1. **必须用标准 CSV 转义**（含换行/逗号/引号的字段整体加双引号，内部引号翻倍为 `""`）。
   不要用「按行 split」或手工拼接字符串来生成/解析——多行字段会把按行读取直接读坏；
2. 编码 **UTF-8**（不带 BOM，与现有快照一致）；
3. 允许为空（平台上的存量资源还没有正文）。空值导出为空字段，engine 会当作空串；
4. 不要截断正文。engine 侧对长文本有自己的截断策略（编码器按 token 处理），
   导出侧截断会造成不可逆的信息损失。

## 5. 验证方法（请 platform 侧导出后自查，engine 侧也会验）

导出后请确认：

```bash
# 1. 用标准 CSV 解析器能读满行数（而非按行 split）
python -c "
import csv
rows = list(csv.DictReader(open('resources.csv', encoding='utf-8')))
print('行数:', len(rows))
print('有 description 列:', 'description' in rows[0])
print('非空正文数:', sum(1 for r in rows if r.get('description','').strip()))
print('含换行的正文数:', sum(1 for r in rows if '\n' in r.get('description','')))
"
```

预期：行数与 `meta.json` 的 `resources_count` 一致；
「含换行的正文数」应当大于 0（若为 0，说明换行在导出时被吃掉了，
这会让正文退化成一大段，虽然不是致命问题，但说明转义环节没有正确保留内容）。

engine 侧收到新快照后会跑：

```bash
python -m scripts.train_semantic --data-source platform --snapshot-dir dataset/platform_snapshot/<run_id>
```

并在结果里把「有正文」与「无正文」的快照指标做对比，回报给 platform。

## 6. 建议同时确认的两件（低优先级，非本次请求范围）

1. **`files_sha256` 校验**：engine 目前尚未校验 `meta.json` 里的校验和
   （这是 engine 侧待办，不是 platform 的问题）。若 platform 侧方便，
   在导出后保留该校验和即可，engine 会补上校验逻辑。
2. **`metadata` 里的结构化信息**：若资源有结构化扩展字段（视频时长、文章字数、
   难度等级等），放进 `metadata_json` 比塞进 `description` 更有价值——engine 可以
   把它们当数值特征用，而不必从文本里猜。

## 7. 附：收到补导后的 engine 侧计划

1. 用新快照重跑训练与评估（`--data-source platform`），确认语义信号是否真的生效；
2. 重新网格搜索质量融合权重（当前 `0.6/0.0/0.4` 是在无正文的 MOOCCube 上调的，
   有真实正文后最优权重会移动）；
3. 把「有正文 vs 无正文」的指标对比回报 platform，作为该列价值的量化依据。

---

## 相关文档

- 交接契约（engine 侧规范）：`edurec-engine/docs/platform-contract.md`
- 设计文档（模型如何用正文）：`edurec-engine/docs/design.md` §3、§4
- 平台侧设计文档存档：`edurec-engine/docs/external/edurec-platform-design.md`
