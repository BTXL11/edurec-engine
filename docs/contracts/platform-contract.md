# engine ↔ platform 交接契约

本文是 **engine 侧必须遵守的规范性文档**（`platform-contract.md`）。两侧之间只有两个方向的数据流动，
除此之外不共享数据库、不共享代码：

```
方向 A  platform ──CSV 快照──► engine      训练与推理的输入，只读
方向 B  engine ──排名 ID 列表──► platform   推理结果，被导入推荐缓存表
```

**核心原则：本契约与模型无关。** platform 完全不感知 engine 用什么算法——它只认「CSV 快照进、
排名 ID 列表出」。因此换模型、换算法、换框架都不应构成破坏性变更；任何模型特有的信息
（分数、召回来源、解释文案、模型名与版本）都放进**旁挂信封**（见「方向 B · 旁挂信封」），
**可选、缺失不报错、平台存而不解释**。

对应地，platform 侧**不运行模型、不做推理**；模型权重只保留在 engine。

**可执行版本**：本契约的 schema、golden fixture 与校验器在
[`../contracts/v1/`](../../contracts/v1/README.md)。改契约时先改那里，再同步 platform 的副本
（`scripts/check-contract-sync.sh`）。

**可选的第三方向**：知识生成 HTTP 服务（`scripts/kg_server.py`）是独立于本契约的
在线能力，见 [`kg-service-contract.md`](kg-service-contract.md)。它不参与推荐主链路。

---

## 方向 A：platform → engine（数据快照）

platform 用 `cmd/export_snapshot` 把业务表导出成一组 CSV，engine 只读消费。

### 目录与命名

```
<platform>/backend/data/snapshots/<run_id>/
```

`run_id` 为导出时刻的**本地时间**，格式 `YYYYMMDD_HHMMSS`（如 `20260915_221100`）。
engine 侧惯例是拷入 `dataset/platform_snapshot/<run_id>/`，目录名保持不变——后续步骤靠它对齐口径。

### 文件清单

| 文件 | 列 | 必需 | 说明 |
|---|---|---|---|
| `users.csv` | `user_id` | ✅ | **只有 ID，不含任何个人信息** |
| `resources.csv` | `resource_id,title,type,category_id,tags_json,metadata_json,avg_rating,view_count,created_at` | ✅ | 见下方 `description` 说明 |
| `resources.csv`（可选列） | `description` | ⬜ | **资源正文**。语义召回模型的全部信号来自它；缺失时 engine 降级为空串（不报错），但**语义模型将基本失效**。建议 platform 在 `export_snapshot` 中补上该列 |
| `behaviors.csv` | `user_id,resource_id,action,ts` | ✅ | 用户行为流水 |
| `ratings.csv` | `user_id,resource_id,score,ts` | ✅ | 站内评分 |
| `meta.json` | 见下 | ✅ | 元信息与校验和。规范快照必须包含；engine 对历史快照缺失时仍兼容（跳过版本/校验和） |
| `categories.csv` | `category_id,name` | ⬜ | 平台会写出；engine 目前不读它（只用 `category_id` 整数，不用分类名） |

> **关于 `description`：** 它不是「新增字段」而是「已有字段的导出」。平台
> `resources` 表本就有 `description TEXT`（见 platform 的 `docs/design/model-design.md` §5.2），
> `export_snapshot` 现已导出该列。engine 侧保持向后兼容：读到就用，
> 读不到就用空串——因此旧快照仍能跑通，只是语义模型会基本失效。

「必需」= 缺了就必须明确报错退出，不要静默继续。

### `meta.json`

```json
{
  "contract_version": 1,
  "run_id": "20260915_221100",
  "exported_at": 1757945471,
  "behavior_window": { "start": 0, "end": 1757945000 },
  "users_count": 1024,
  "resources_count": 2380,
  "behaviors_count": 51200,
  "ratings_count": 3104,
  "categories_count": 12,
  "files_sha256": { "users.csv": "…", "resources.csv": "…", "categories.csv": "…",
                    "behaviors.csv": "…", "ratings.csv": "…" }
}
```

- `contract_version`：**当前为 `1`**。engine 读到不认识的版本应当**明确报错退出**，不要猜测字段含义。
- `exported_at`、`behavior_window.*`、`created_at`、`ts` **一律是 Unix 秒**（不是毫秒、不是字符串）。
- `behavior_window.start` 当前恒为 `0`（即导出全量行为，不按时间窗裁剪）；`end` 是导出时见到的最大时间戳。
- `files_sha256` 用于校验拷贝完整性：engine 加载前会逐个比对，不匹配直接报错（挡住拷贝中断）。

### 取值约定

- 分隔符为 `,`，字段按需加引号（标准 CSV）。文本内若含逗号/引号/换行，由 CSV 转义处理，**不要自己拼字符串**。
- 编码 **UTF-8**。
- `avg_rating` 固定两位小数的十进制字符串（如 `4.50`）。
- `tags_json`、`metadata_json` 是 **JSON 文本**（数组 / 对象），不是分隔符字符串；`tags` 为空时导出为 `[]`。
  平台保证这两个字段是合法 JSON；当前 loader 对 `metadata_json` 有容错、对 `tags_json` **没有**，手工改过快照时会崩。
- 每份 CSV **按主键升序**导出，顺序稳定，但不构成语义——需要顺序请自行排序。
- ID 可能有空洞、可能从 `0` 开始（导出的是平台真实主键，不重编号）。**不要假设 ID 连续**。

### engine 的义务

- **只读**。不得向快照目录写入、不得修改快照文件；需要中间产物请写自己的目录。
- 用 `meta.json` 判版本、校验和，而不是靠文件名或列数猜。
- 通过既有的 CLI 入口消费，保证训练与推理口径一致：

  ```bash
  python -m scripts.train_semantic --data-source platform --snapshot-dir dataset/platform_snapshot/<run_id>
  python -m scripts.run_batch_infer --data-source platform --snapshot-dir dataset/platform_snapshot/<run_id>
  ```

  `run_batch_infer` 的两个参数也可用 `ENGINE_DATA_SOURCE` / `ENGINE_SNAPSHOT_DIR` 环境变量给，
  但**显式传参优先**——两条命令用同一份 `--snapshot-dir` 是「训练与推理口径一致」的前提。

- **`--data-source platform` 是唯一会被导入 platform 的轨道。** sim 等自造数据轨仅用于
  演示与回归测试，其 ID 与平台库无关，**不得**把它们的输出导入 platform。

---

## 方向 B：engine → platform（推荐结果）

engine 推理完成后写出结果文件，由人工或脚本拷到 platform，再经管理接口导入推荐缓存表。

### 位置与命名

| 环节 | 路径 |
|---|---|
| engine 写出 | `<engine>/model/recommendations.json` |
| 拷入 platform | `<platform>/backend/data/recommendations.json` |

platform 侧的实际路径由配置项 `engine.recommendations_file` 决定（默认即上表值，相对 `backend/`）。
拷贝动作目前是手动的（`scripts/handoff.sh` 第 ⑤ 步），engine 不需要知道 platform 的目录布局。

> engine 侧默认写到 `cfg.model_dir`（默认 `"model"`），相对**当前工作目录**解析；
> `run_batch_infer` 可用 `--out-dir` 覆盖产物目录、`--model-dir` 覆盖模型目录。
> `handoff.sh` 使用默认路径，因此它要求**在 engine 仓库根目录运行推理**——
> 否则产物会落在别处，而 platform 的拷贝步骤仍去 `model/` 找它。

### 主文件格式（**逐字节严格**）

```json
{
  "1": [42, 17, 305],
  "2": [11, 7, 30],
  "5": [3]
}
```

类型就是 JSON 对象 → 字符串键 → 非负整数数组，等价于 Go 的 `map[string][]uint`。platform 侧
**不做任何容错解析**，以下任一条不满足都会让**整份文件**导入失败（HTTP 500，一条都不写入）：

| 硬约束 | 说明 |
|---|---|
| 顶层只能是这个对象 | **不得**添加 `contract_version`、`generated_at` 等任何兄弟字段——值不是整数数组就会解析失败 |
| 键必须是十进制无符号整数 | 即平台 `user_id`。**非数字键会被静默忽略**（不报错，但那个用户不会更新） |
| 值必须是 JSON 数组 | 数组元素必须是**数字**形式的非负整数；写成字符串 `"42"`、负数、小数（`42.0`）都会解析失败 |
| 必须是合法 UTF-8 且**无 BOM** | 带 BOM 会解析失败。Python 用 `encoding="utf-8"` 而非 `utf-8-sig` 写 |
| 必须是一次写完整的文件 | 建议**先写临时文件再原子重命名**，避免 platform 读到写了一半的内容 |

### 语义约定

- **数组顺序就是排名**，最相关的在前。platform 端到端保序（导入保序、读取时按缓存顺序重排），
  不会重排、不会打分、不会截断——除展示层 `limit` 外不做任何裁剪。
  展示层 `limit` 默认 20、上限 50。
- **必须自行去重**。platform 不去重，同一用户列表里的重复 ID 会原样进入缓存与返回结果。
- **覆盖全量用户**，包括冷启动用户（平台上无行为的用户也应出现在结果里，走热门兜底）。
- **用户缺失、或该用户为空数组 `[]`**：platform 视为「本次不更新该用户」，**保留其旧缓存行**，
  不会清空。因此不必为了「清空」而输出空数组——它表达不了「清空」。
- 用户 ID 在平台库中不存在 → 该用户被跳过，旧缓存保留，计入 `skipped_users`。
  某用户的所有资源 ID 都不在平台库中 → 同样跳过该用户。
- 资源 ID 不在平台库中 → 只从该用户的列表中剔除，其余资源照常导入，计入 `skipped_resources`。
- **导入不是跨用户原子的**：platform 逐用户「删旧行 + 写新行」，中途失败时前面已写入的用户不会回滚。
  文件本身若是完整的，通常不会走到这里；这条主要用于解释「为什么部分用户更新了、部分没有」。

### 导入接口

```http
POST /api/v1/admin/recommendations/import
Authorization: Bearer <管理员 access_token>
```

无请求体参数，读取配置指向的文件。返回统计：

```json
{ "imported_users": 1024, "skipped_users": 0,
  "imported_resources": 204800, "skipped_resources": 12 }
```

导入是**幂等**的：同一份文件重复导入结果相同，缓存整行被覆盖。

### 旁挂信封（`recommendations.meta.json`，**可选**）

因为主文件顶层**不能**加字段，任何元信息只能另开一个文件。约定：

```
主文件   <platform>/backend/data/recommendations.json
信封文件 <platform>/backend/data/recommendations.meta.json
```

规则：与主文件**同目录**，文件名为主文件名去掉扩展名后加 `.meta.json`。
建议 engine 在**同一目录**下与主文件一起写出。

```json
{
  "contract_version": 1,
  "generated_at": 1757946232,
  "run_id": "20260915_221100",
  "snapshot_run_id": "20260915_091500",
  "model": { "name": "dssm+deepfm", "version": "v1" },
  "method": { "recall": "dssm", "rank": "deepfm", "rerank": "mmr" },
  "top_n": 200,
  "users_count": 1024,
  "scores":  { "2": [0.93, 0.87, 0.51] },
  "reasons": { "2": ["与你学过的《线性代数》相关"] }
}
```

| 字段 | 说明 |
|---|---|
| `contract_version` | 信封自身的版本，当前 `1`。**唯一建议必填**的字段，便于将来演进 |
| `generated_at` | 本次推理完成时刻，**Unix 秒**。平台目前只能把「导入时刻」当作 `updated_at` 返回给前端，因此这是唯一能说明结果真实新鲜度的来源 |
| `run_id` / `snapshot_run_id` | 本次推理的标识 / 所用快照的 `run_id`，用于追溯「这批结果是基于哪份数据、可复现」 |
| `model.name` / `model.version` | 模型标识。**换模型时这里变，主文件格式不变** |
| `method.*` | 召回/精排/重排各自的方法名，纯描述性 |
| `top_n` | 每用户输出的最大条数 |
| `scores` | 每用户的分数，**顺序与主文件数组一一对应**。供未来重排/展示/离线评估使用 |
| `reasons` | 可读的解释文案，**同一位置一一对应**。供未来做「为什么推荐这个」 |

**约束**：

- 信封是**可选**的。缺失、为空、字段不全都**不得**影响平台导入，平台忽略未知字段。
- 平台**存储但不解释**这些字段：不因 `scores` 改变排序，不因 `model.name` 改变行为。
- 将来若新模型不产生分数或解释，**只是不写这两个字段**，不构成契约变更。
- `scripts/handoff.sh` 第 ⑤ 步会同时拷贝主文件与信封；信封缺失时只拷主文件，不影响导入。

---

## 换模型时的准则

这是本契约存在的理由。换掉 `dssm+deepfm+mmr` 里的任何一个环节时：

**允许且不需要协调的（不构成契约变更）**

- 换召回/精排/重排算法，或整个换成别的范式（协同过滤、图模型、LLM 重排……）
- 换模型结构、特征、训练目标、超参
- 改 `top_n`
- 在信封里增删任何可选字段，或干脆不写信封
- 提高结果质量导致排名变化

**属于契约变更、必须两侧协调的**

- 改动主文件的形状（键或值的类型、嵌套层级）——会直接让 platform 报 500
- 改变「平台 ID」这一语义（例如改成输出引擎内部编号）
- 改变顺序语义（例如改成无序集合）
- 引入新的交接物（新文件、新接口）

**唯一必须始终保持的**

1. 主文件是 `{平台 user_id: [平台 resource_id, …]}`，按相关性降序、已去重、覆盖全量用户；
2. ID 取自所用快照的 `users.csv` / `resources.csv`，不引入任何引擎内部编号；
3. 遵守 `meta.json` 的 `contract_version`，读到不认识的版本明确报错。

只要这三点不动，换任何模型，platform 一行代码都不用改。

---

## 故障对照表

| 平台侧症状 | 原因 | 修法 |
|---|---|---|
| 导入返回 500 | 主文件顶层有值不是整数数组的字段（如 `version`） | 元信息移入旁挂信封 |
| 导入返回 500 | 资源 ID 写成字符串（`"42"`）或小数（`42.0`） | 输出 JSON 整数 |
| 导入返回 500 | 文件带 UTF-8 BOM | 用 `encoding="utf-8"` 写，不要 `utf-8-sig` |
| 导入返回 500 | 非法 UTF-8 / 文件被截断 | 检查写出过程，改为原子重命名 |
| `skipped_users` 异常大 | 键不是平台用户 ID，或用了 sim 轨道的输出 | 确认用 `--data-source platform` 且 ID 取自快照 |
| `skipped_resources` 异常大 | 输出了快照 `resources.csv` 里没有的资源 ID | 以快照为准过滤 |
| 某些用户推荐不更新 | 该用户在文件里缺失，或被判为空数组 | 覆盖全量用户；空数组表达不了「清空」 |
| 前端看到的 `updated_at` 是刚刚 | 平台把「导入时刻」当作 `updated_at` | 真实生成时间写进信封的 `generated_at` |
| 列表顺序与预期不符 | 平台按数组顺序当作排名，不做重排 | 按相关性降序输出 |
| 同一资源重复出现 | 平台不去重 | 在 engine 出口去重 |

---

## 合规与安全

- 快照中的 `users.csv` **只有用户 ID，不含任何个人信息**；engine 侧也不得把个人信息写回任何交接物。
- 交接物只包含「用户 ID → 资源 ID → 分数/解释」这类派生数据，不含原始行为明细的回流。
- 交接通过**文件**完成，两仓库目录互不读写；engine 不需要、也不应该直连 platform 的数据库或 HTTP 接口
  （导入动作由 platform 的管理员接口发起）。
- 唯一例外是**可选的**知识生成 HTTP 服务（方向 C）：platform 调用 engine 的
  `kg_server.py`，engine 不反向调用 platform。它不参与推荐主链路，契约见
  [`kg-service-contract.md`](kg-service-contract.md)。
- 模型权重、特征表、训练日志等一律**不进入**交接物，只保留在 engine。

---

## 附录：engine 现状与本契约的差距

> 以下是 **2026-11 的核对结果**（契约 v1 + `contracts/v1` 落地后）。
> 前面的规范部分不随实现变化；本清单不是契约，修掉差距不构成契约变更。

### 已经符合的（换模型时必须保持）

| 条款 | 现状 |
|---|---|
| 主文件形状 | ✅ `{str(user_id): [int resource_id, …]}`，`encoding="utf-8"`（无 BOM） |
| 用平台原始 ID | ✅ **平台轨道上 ID 原样透传**：`platform.py` 直接用快照里的整数 ID，不做任何重编号。MOOCCube 轨道不同——课程 ID 是字符串，加载器排序后编号，两条轨道**不可相互套用** |
| 覆盖全量用户 | ✅ `infer_batch` 默认遍历 `bundle.users` 全部用户，冷启动走热门兜底并有计数 |
| 按相关性降序 | ✅ 语义分与质量分融合后降序，顺序即排名 |
| 自行去重 | ✅ 出口按 `argsort` 下标取候选，天然唯一；测试断言无重复 |
| 硬约束「先写临时文件再原子重命名」 | ✅ `write_handoff` 写 `*.tmp` 后 `os.replace` |
| 旁挂信封 | ✅ 写出 `recommendations.meta.json`，含 `scores`、模型标识、质量权重、`run_id` / `snapshot_run_id`；`--no-envelope` 可关闭 |
| 推荐里不含已交互内容 | ✅ 出口按 `behaviors` 过滤该用户已交互的课程 |
| 训练/推理口径一致 | ✅ 推理前校验编码器版本**与资源语料指纹**，不一致时明确报错退出 |
| 加载前校验 `files_sha256` | ✅ `platform.py` 在 `meta.json` 带校验和时逐个比对，不匹配直接 `ValueError` |
| 出口自检硬失败 | ✅ `run_batch_infer` 调用 `contracts/v1/validate.py`，违规即 `SystemExit`，不依赖 `assert` |
| 数据源不会错选 | ✅ `--data-source` 在训练/推理入口均**必填**，不再默认 `mooccube` |
| 资源正文 `description` | ✅ `export_snapshot` 已导出该列；缺失时 engine 仍降级为空串 |

### 仍需注意（不构成契约违规）

| 项 | 现状 | 说明 |
|---|---|---|
| `categories.csv` | engine 不读取 | 只用 `category_id` 整数，保留该文件便于人工核对 |
| 产物路径 | 默认固定单槽位 `model/`；`run_batch_infer` 已支持 `--out-dir` | platform 约定读单一文件；切换数据源轨道时用 `--out-dir` 或先备份 |
| 信封 `generated_at` | ✅ 已写；`run_id` 也已写 | 平台用它展示真实新鲜度 |
| 多路召回 | 当前单路语义 + 质量融合 | 一旦合并多路候选，必须在出口补显式去重 |
| `metadata_json.created_at` | 已读入 metadata，未用于特征 | 新资源加权属于后续模型工作 |

### 接入前应当知道

- **这一版模型的收益尚未被证明。** MOOCCube 上语义召回的 HitRate@10 = 0.4432，
  低于热门基线 0.4930，NDCG 与 MRR 同样落后；融合热度后 HitRate@10 = 0.5758 才超过热门。
  数字与诊断见 [`gaussian-recall-status.md`](../design/gaussian-recall-status.md)。
  「契约通过」只说明能接得上，不等于值得上线。
- **platform 侧数据量是真正的瓶颈。** 快照规模小（用户/行为少、资源以视频为主）时，
  任何模型都训练与评估不出结论；先积累真实行为，再谈模型收益。

### 已解决的历史问题（保留记录，便于对照）

- ✅ 去重曾是隐式的、`rerank` 的「已看不推」曾从未生效、冷启动加权曾是均匀缩放：
  旧流水线（MMR 重排那一版）的问题，随模型重建一并消失。
- ✅ `movielens` 轨道已移除：入口只支持 `mooccube` / `sim` / `platform`。
- ✅ `scripts.train_all` / `scripts.infer_batch` 是旧文档里的错误命令名，已统一为
  `train_semantic` / `run_batch_infer`。
- ✅ `export_snapshot` 曾不导出 `description`，现已补齐（见「方向 A」）。

---

## 相关文档

- 契约的可执行版本：[`../contracts/v1/README.md`](../../contracts/v1/README.md)
- 可选在线能力：[`kg-service-contract.md`](kg-service-contract.md)
- 当前数据流：[`data-flow.md`](data-flow.md)
- platform 侧：[`docs/contracts/engine-integration.md`](https://github.com/Shionyori/edurec-platform/blob/main/docs/contracts/engine-integration.md)
  ——接入形态与演进方向；[`docs/guides/data-handoff.md`](https://github.com/Shionyori/edurec-platform/blob/main/docs/guides/data-handoff.md)
  ——一轮刷新的完整步骤
