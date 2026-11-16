# 数据流（当前实现）

本文只描述**现在**跑得通的链路。规范性约束以
[`platform-contract.md`](platform-contract.md) 为准；机器可读版本与 golden fixture 在
[`../contracts/v1/`](../../contracts/v1/README.md)。

```
platform 业务库 (MySQL，唯一事实来源)
   │  cmd/export_snapshot
   ▼
数据快照 <run_id>/  users.csv / resources.csv / behaviors.csv / ratings.csv
   │                categories.csv / meta.json（contract_version + sha256）
   │  拷入 engine dataset/platform_snapshot/<run_id>/
   ▼
edurec-engine
   │  scripts/train_semantic.py  → model/semantic_recall.pt
   │  scripts/run_batch_infer.py → model/recommendations.json
   │                               model/recommendations.meta.json（可选信封）
   ▼
platform backend/data/recommendations.json
   │  POST /api/v1/admin/recommendations/import
   ▼
recommendations 缓存表 → GET /api/v1/recommendations（按数组顺序返回）
```

## 一轮刷新

```bash
# platform 仓库根目录（engine 默认在 ../edurec-engine）
bash scripts/handoff.sh               # 导出 → 校验快照 → 训练 → 推理 → 校验产物 → 导入
bash scripts/handoff.sh --infer-only  # 复用已有模型，只导出 → 推理 → 导入
```

`handoff.sh` 的每一步与手动命令一一对应，并在快照/产物交接前调用
`contracts/v1/validate.py`；任何一步不符合契约都会中止。

手动等价命令：

```bash
# ① platform：导出快照
cd edurec-platform/backend
CONFIG_PATH=configs/config.yaml go run ./cmd/export_snapshot

# ② 校验并拷入 engine
python edurec-engine/contracts/v1/validate.py snapshot \
  data/snapshots/<run_id>
cp -r data/snapshots/<run_id> ../edurec-engine/dataset/platform_snapshot/

# ③ engine：训练 + 推理（必须在 engine 仓库根目录运行）
cd ../edurec-engine
.venv/bin/python -m scripts.train_semantic \
  --data-source platform --snapshot-dir dataset/platform_snapshot/<run_id>
.venv/bin/python -m scripts.run_batch_infer \
  --data-source platform --snapshot-dir dataset/platform_snapshot/<run_id>

# ④ 校验产物并回传
.venv/bin/python contracts/v1/validate.py recommendations \
  model/recommendations.json --envelope model/recommendations.meta.json
cp model/recommendations.json ../edurec-platform/backend/data/recommendations.json
cp model/recommendations.meta.json ../edurec-platform/backend/data/recommendations.meta.json

# ⑤ platform：管理员登录后导入
# POST /api/v1/admin/recommendations/import
```

## 交接物

### 方向 A：platform → engine（快照）

| 文件 | 必需 | 说明 |
|---|---|---|
| `users.csv` | ✅ | 只有 `user_id`，不含个人信息 |
| `resources.csv` | ✅ | `resource_id,title,description,type,category_id,tags_json,metadata_json,avg_rating,view_count,difficulty,duration_minutes,created_at` |
| `behaviors.csv` | ✅ | `user_id,resource_id,action,ts`（action ∈ view/click/favorite） |
| `ratings.csv` | ✅ | `user_id,resource_id,score,ts`（score 1–5） |
| `categories.csv` | ⬜ | `category_id,name`；engine 不读，保留用于核对 |
| `meta.json` | ✅ | `contract_version=1`、`run_id`、时间戳、行数、`files_sha256`（规范快照必须；旧快照缺失时兼容） |

- 时间戳一律 Unix 秒；编码 UTF-8、无 BOM；CSV 按主键升序。
- engine 加载前会校验 `contract_version` 与 `files_sha256`；校验失败明确报错。
- `description` 是语义模型的**全部信号来源**；缺失时降级为空串，但模型基本失效。

### 方向 B：engine → platform（推荐结果）

主文件 `recommendations.json`：

```json
{"1": [103, 102], "2": [101]}
```

- 顶层**只能**是 `{平台 user_id: [平台 resource_id, …]}`，无兄弟字段；
- 键/值都是十进制非负整数；数组顺序即排名；必须已去重；
- 覆盖快照全量用户（冷启动走热门兜底）；空数组表示「本次不更新该用户」；
- 先写 `.tmp` 再原子重命名。

旁挂信封 `recommendations.meta.json`（可选，缺失不影响导入）：

```json
{
  "contract_version": 1,
  "generated_at": 1767225700,
  "run_id": "20260101_000100",
  "snapshot_run_id": "20260101_000000",
  "model": {"name": "semantic_two_tower", "version": "v1", "encoder": "…"},
  "top_n": 20,
  "users_count": 3,
  "scores": {"1": [0.93, 0.62]},
  "reasons": {"1": ["与你学过的《线性代数》相关", "…"]}
}
```

### 方向 C：platform → engine（可选在线概念生成）

`scripts/kg_server.py` 提供 `POST /knowledge/generate`，平台在「自动生成概念」时调用。
这不是主链路，engine 未启动只影响该功能。接口与版本规则见
[`kg-service-contract.md`](kg-service-contract.md)。

## ID 语义（不要混用）

| 数据域 | ID 形态 | 出现位置 |
|---|---|---|
| platform / B 站 | 平台主键，整数 | 快照 CSV、推荐结果、`resources.id` |
| MOOCCube | 字符串（`C_course-v1:…`） | `dataset/MOOCCube/`、研究脚本 |

只有 `--data-source platform` 的产物可以导入 platform。sim / MOOCCube 产物
导入后 `skipped_users` 会接近全量，而导入接口只回报计数、不报错。

## 失效与降级

| 情况 | 行为 |
|---|---|
| 快照缺 `description` | engine 降级为空串，语义模型基本失效（不报错） |
| 快照 `contract_version` 不认识 | engine 明确报错退出 |
| 快照 sha256 不匹配 | engine 明确报错退出（挡住拷贝中断） |
| 推荐产物不符合契约 | `run_batch_infer` / `handoff.sh` 中止，不交付 |
| 信封缺失/损坏 | 平台照常导入主文件，只是没有 run/理由元信息 |
| 用户在结果里缺失或为空数组 | 平台保留旧缓存行 |
| 用户/资源 ID 不在平台库 | 逐项跳过并计数 |
| 推荐文件为空/损坏 | 平台导入失败或空结果，缓存不变，服务继续兜底 |
