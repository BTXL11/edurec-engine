# 数据传输过程文档

记录本项目涉及的数据从采集、落库、交接、推理到回传的完整流转。所有结论都基于两侧仓库的实际代码与数据文件核对，
不成立的推断已标注为「待确认」。

相关文档：
* [`edurec-engine/docs/platform-contract.md`](file:///E:/work/learn/edurec-engine/docs/platform-contract.md) —— 平台↔引擎的**规范性交接契约**（权威）
* [`recommender/README.md`](recommender/README.md) —— 本仓库的推荐原型
* [`VISUALIZATION.md`](VISUALIZATION.md) —— 知识图谱可视化

---

## 一、数据分层

```
┌─────────────────────────────────────────────────────────────────────┐
│ L1 数据源（B 站）                                                    │
│   搜索接口 / 视频详情 / 评论                                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  HTTP 采集（关键词搜索 + 详情补全）
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ L2 采集器                                                            │
│   edurec-platform/backend/crawler/        （JSON 输出）              │
│   ER-backend/bilibil_keyword_Crawler/     （写 bili_videos 等表）     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  JSON 落文件（data/bilibili/latest.json）
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ L3 导入（Go）                                                        │
│   cmd/import_bilibili  → service.CrawlImportService                  │
│   校验：type 枚举 / 允许分区 / 内容规则 → 写 resources 表             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  GORM 写入
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ L4 MySQL（platform 业务库，**唯一事实来源**）                          │
│   resources   /  user_behaviors  /  users  /  categories  /  ratings  │
└───────────┬──────────────────────────────────┬──────────────────────┘
            │                                  │
     ①正式交接导出                        ②数据库直接导出
   cmd/export_snapshot                   （本项目根目录的两个 CSV）
            │                                  │
            ▼                                  ▼
┌───────────────────────────┐      ┌──────────────────────────────────┐
│ L5 快照（契约格式）         │      │ L5′ 表导出（完整行）               │
│ data/snapshots/<run_id>/   │      │ search-实变函数.csv (37×17)       │
│  resources.csv   9~12 列    │      │ user_behaviors.csv   (59×7)      │
│  behaviors.csv   4 列       │      │ 带 updated_at / deleted_at        │
│  users.csv   ratings.csv   │      │ 时间戳为 ISO 字符串                │
│  meta.json（SHA256）       │      └──────────────────────────────────┘
└───────────┬───────────────┘
            │  只读消费（engine 不写快照目录）
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ L6 推荐引擎                                                          │
│   edurec-engine  --data-source platform --snapshot-dir <dir>         │
│   语义召回训练 / 批量推理                                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  写出 model/recommendations.json
                           │  ＋ recommendations.meta.json（旁挂信封）
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ L7 回传（人工拷贝 → 管理接口导入）                                     │
│   backend/data/recommendations.json                                  │
│   POST /api/v1/admin/recommendations/import                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ L8 在线服务                                                          │
│   GET 推荐接口 → 按缓存数组顺序返回（保序、不重排、不截断）             │
│   埋点写回 user_behaviors 表 → 回到 L4                               │
└─────────────────────────────────────────────────────────────────────┘
```

**闭环**：L8 的行为埋点写回 L4，下一轮快照（L5）再带上新行为 —— 这就是上面 `behaviors` 从 37 涨到 59 的原因。

---

## 二、每种交接物的字段与格式

### 2.1 `resources` 表（L4 的权威结构）

`backend/internal/model/resource.go`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | uint | 主键（`gorm.Model`） |
| `created_at` / `updated_at` / `deleted_at` | time | 审计列（`gorm.Model`），`deleted_at` 为 NULL 表示未软删除 |
| `title` | varchar(256) | 标题 |
| `description` | text | **资源正文** —— 语义召回的全部信号来源 |
| `cover_url` | varchar(512) | 封面 |
| `type` | varchar(20) | `course` / `article` / `video`（枚举受 `IsValidResourceType` 校验） |
| `category_id` | uint | 分类外键 |
| `tags` | json | JSON 数组字符串 |
| `metadata` | json | 扩展字段（B 站来源的 `bvid`/`like`/`favorite`/`pubdate`/`duration`/`typename` 放这里） |
| `author` | varchar(128) | UP 主 / 作者 |
| `source_url` | varchar(512) | 原始链接，**用于去重与溯源** |
| `avg_rating` | decimal(2,1) | 站内均分 |
| `view_count` | uint | 播放量 |
| `difficulty` | varchar(16) | `beginner`/`intermediate`/`advanced`，空串表示未知 |
| `duration_minutes` | uint | 时长（分钟），0 表示未知 |

### 2.2 `user_behaviors` 表

`backend/internal/model/user_behavior.go`：

| 字段 | 说明 |
| --- | --- |
| `id` / `created_at` / `updated_at` / `deleted_at` | `gorm.Model` 审计列 |
| `user_id` | 外键 → `users.id` |
| `resource_id` | 外键 → `resources.id`（**B 站侧资源 ID**） |
| `action` | `view` / `click` / `favorite` |

### 2.3 快照格式（L5，契约格式）

`cmd/export_snapshot/main.go` 决定列与格式：

| 文件 | 列 | 时间戳 |
| --- | --- | --- |
| `users.csv` | `user_id`（**只有 ID，不含任何个人信息**） | — |
| `resources.csv` | `resource_id,title,description,type,category_id,tags_json,metadata_json,avg_rating,view_count,difficulty,duration_minutes,created_at` | Unix 秒 |
| `behaviors.csv` | `user_id,resource_id,action,ts` | Unix 秒 |
| `ratings.csv` | `user_id,resource_id,score,ts` | Unix 秒 |
| `categories.csv` | `category_id,name` | — |
| `meta.json` | `contract_version=1`、`run_id`、`exported_at`、`behavior_window`、各表行数、`files_sha256` | Unix 秒 |

目录命名：`data/snapshots/<YYYYMMDD_HHMMSS>/`（**本地时间**）。

契约要点（详见 `platform-contract.md`）：
* 分隔符 `,`、编码 **UTF-8**（不带 BOM），字段按需加引号，标准 CSV 转义
* `avg_rating` 固定两位小数；`tags_json`/`metadata_json` 是 **JSON 文本**，`tags` 空时写 `[]`
* 每份 CSV 按主键升序；**ID 可能有空洞，不要假设连续**
* engine **只读**快照；`contract_version` 不认识就明确报错退出
* `description` 是**可选列**：缺失时 engine 降级为空串不报错，但**语义模型基本失效**

### 2.4 表导出格式（L5′，本项目根目录的两个文件）

这是**数据库表的完整行导出**，不是契约快照：

| 文件 | 行×列 | 对应表 | 列 |
| --- | --- | --- | --- |
| `search-实变函数.csv` | 37 × 17 | `resources` | 2.1 节全部 17 列 |
| `user_behaviors.csv` | 59 × 7 | `user_behaviors` | 2.2 节全部 7 列 |

与快照的关键差异：

| 维度 | 快照（L5） | 表导出（L5′） |
| --- | --- | --- |
| 列 | 收敛为 engine 需要的子集（9~12 列） | 表全列（17 / 7 列），含 `updated_at` / `deleted_at` |
| 时间戳 | Unix 秒（整数） | ISO 字符串（`2026-09-15 21:45:09`） |
| 审计列 | 不导出 | `deleted_at` 导出为字面量 `\N`（MySQL `SELECT ... INTO OUTFILE` 风格的空值表示） |
| 校验 | `meta.json` 带 `files_sha256` | 无 |
| 命名 | 固定文件名 + `run_id` 目录 | 自由（按用途命名，如 `search-<关键词>.csv`） |
| 用途 | engine 训练/推理输入 | 人工核查、按关键词取样本 |

> **注意**：`search-实变函数.csv` 里的 37 条**都是 `resources` 表的行**，只是按"实变函数"这个关键词筛出来的一批，
> 它**不是**一个独立的搜索日志表，也**不是**搜索引擎返回结果的原始响应。

---

## 三、实测数据对齐（2026-10-13 核对）

用 `edurec-platform/backend/data/snapshots/` 下的快照与本项目两个 CSV 逐项对比：

| 快照 `run_id` | users | resources | behaviors | ratings | 说明 |
| --- | --- | --- | --- | --- | --- |
| `20260910_203538` / `_205745` / `_212055` | 2001 | 500 | 100000 | 9840 | sim 轨（模拟数据），三份内容相同 |
| `20260916_181925` | 2 | 139 | 14 | 0 | platform 轨早期真实数据 |
| `20260919_115713` | 2001 | 500 | 100017 | 9840 | sim 轨 |
| `20260919_115858` | 2002 | 518 | 100017 | 9840 | sim 轨 |
| **`20260919_121613`** | **4** | **170** | **37** | **3** | **platform 轨真实数据 ← 与本项目 CSV 同源** |

对齐结果（以 `20260919_121613` 为基准）：

| 比对项 | 结果 |
| --- | --- |
| `search-实变函数.csv` 的 37 个 id | 9 个落在该快照内，**标题 9/9 逐字一致** |
| 该快照的 37 条 behaviors | (user_id, resource_id, action) 三元组 **34/34 全部命中** `user_behaviors.csv` |
| `user_behaviors.csv` 多出的 15 行 | 快照之后新增的行为（用户继续产生了观看/收藏） |
| 快照后新增的 28 个资源 | id 627~801，即后续几轮采集新导入的课程 |
| 用户集合 | 快照为 `[2, 3, 2000, 2001]`；本项目 CSV 只涉及 `[2, 3]` |

**结论**：本项目根目录的两个 CSV 是 **platform 轨真实业务数据的表导出**，与 `20260919_121613` 快照同源，
时间上晚于该快照（数据更全）。

---

## 四、两处数据质量缺口

### 4.1 `description` 列缺失 —— 会让语义召回失效

`description`（资源正文）是语义召回模型的**全部信号来源**。契约文档明确记录：
`export_snapshot` 早期未导出该列，缺失时 engine 降级为空串、不报错，但**语义模型基本失效**。

现状核对：
* 当前代码 `resourceHeader()` 已包含 `description`（12 列）
* 但 `data/snapshots/` 下**所有快照仍是旧的 9 列**，即历史快照都不含正文
* 本项目 `search-实变函数.csv` 作为**表导出是带 `description` 的**（37 行全部非空）

**该做**：重新执行一次 `cmd/export_snapshot`，让快照带上 `description`；之后训练/推理才有效。

### 4.2 `difficulty` / `duration_minutes` 全空

这两列是**教育推荐区别于通用推荐的核心结构化特征**（难度匹配、时长匹配）。实测：

| 文件 | difficulty | duration_minutes |
| --- | --- | --- |
| `search-实变函数.csv`（37 行） | 全部为 `\N`（空） | 全部为 `0` |
| 快照 `20260919_121613` | 该列不存在（旧格式） | 该列不存在 |

也就是说这两列**从未被真正填充过**。而上游其实拿得到时长 —— B 站原始数据里有：
* `metadata.duration`：形如 `"3:55"`、`"1866:37"`（字符串）
* `metadata.typename`：形如 `"校园学习"`（可辅助推断难度）

**该做**：在 `CrawlImportService` 落库时把 `metadata.duration` 解析成分钟写入 `duration_minutes`；
`difficulty` 需要一套规则或模型（可从标题关键词、分区、时长综合判断）。

---

## 五、ID 体系（**不要把两套 ID 混用**）

| 数据域 | ID 形态 | 例子 | 出现在 |
| --- | --- | --- | --- |
| **B 站 / platform** | 纯整数，平台主键 | `resources.id` = `59`、`694`、`801`<br>`users.id` = `2`、`3` | `search-实变函数.csv`、`user_behaviors.csv`、快照 CSV、`recommendations.json` |
| **MOOCCube** | 字符串，带前缀 | `C_course-v1:TsinghuaX+30240184+sp`<br>`U_7001215` | `data/MOOCCube/*`、本项目的 `.ttl` 文件名、`rec_cache/` |

两套体系**互不相通**：
* B 站侧的课程在 MOOCCube 里没有对应条目，反之亦然
* platform 轨的 `resource_id` 与 MOOC 轨的课程 ID **不可互相映射**
* 因此本项目 `recommender/` 的两条轨道处理方式不能相互套用 ——
  `platform` 轨的 ID 原样透传（整数），`mooccube` 轨的课程 ID 是字符串、需排序后重新编号，
  原始 ID 保留在 `metadata["source_id"]` 与 `id_map` 里

> 契约文档里记了一条真实事故：`run_batch_infer` 的 `--data-source` **默认值是 `mooccube`**。
> 不带参数直接跑，产出的 `recommendations.json` 里装的是 MOOC 的 ID，拷进 platform 导入时
> `skipped_users` 接近全量 —— 而导入接口**只回报跳过计数、不报错**，现象是「导入成功但推荐没变化」。
> 所以只要产物要交给 platform，**必须显式带 `--data-source platform --snapshot-dir <dir>`**。

---

## 六、schema 漂移注意点

`data/snapshots/` 下的快照是**旧版本代码**产出的：

| 项 | 旧快照（实际文件） | 当前代码（`resourceHeader()`） |
| --- | --- | --- |
| `resources.csv` 列数 | **9** | **12**（多了 `description`、`difficulty`、`duration_minutes`） |
| `behaviors.csv` | 4 列，一致 | 4 列 |
| `meta.json` | 无 `description` 相关 | `contract_version=1` 不变 |

契约里 `description` 被定义为「**可选列**，补上不构成破坏性变更」，所以这次扩列是安全的；
但**旧快照不要直接拿去跑新模型**，否则正文为空、语义召回失效（见 4.1）。

---

## 七、检查清单（一轮数据交接）

| # | 动作 | 命令 / 位置 | 验证点 |
| --- | --- | --- | --- |
| 1 | 采集 | `backend/crawler/run.py` | `data/bilibili/latest.json` 有新增 |
| 2 | 导入 | `cmd/import_bilibili -file <json>` | 输出新增/刷新/跳过计数；先 `--dry-run` |
| 3 | 导出快照 | `cmd/export_snapshot` | 新 `<run_id>` 目录；`resources.csv` 有 **12 列**且 `description` 非空 |
| 4 | 校验快照 | `meta.json` | `contract_version=1`、行数合理、`files_sha256` 齐 |
| 5 | 推理 | `--data-source platform --snapshot-dir <dir>` | 在 **engine 仓库根目录**运行（产物路径不可覆盖）；无 BOM、UTF-8 |
| 6 | 拷贝 | `backend/data/recommendations.json` | 顶层只能是 `{user_id: [resource_id]}`，不得有兄弟字段 |
| 7 | 导入 | `POST /api/v1/admin/recommendations/import` | `skipped_users` / `skipped_resources` 不应异常大 |
| 8 | 复核 | 查询推荐接口 | 顺序 = 数组顺序；`updated_at` 是导入时刻，真实生成时间看信封 `generated_at` |

---

## 八、待确认事项

以下是目前**没有证据支撑**、不应写进结论的点：

1. `search-实变函数.csv` 的**具体导出方式**（GUI 工具 / SQL / 后端接口）未确认 ——
   只能确定列结构与 `resources` 表全列一致、`deleted_at` 呈现 `\N` 字面量。
2. 用户只提供了 `resources` 与 `user_behaviors` 两张表的导出，`users` / `ratings` / `categories`
   三张表在本项目内**没有样本**，其导出格式未核对。
3. 快照 `20260919_121613` 的 `ratings.csv` 有 3 条记录，本项目**没有对应文件**，未核对。

---

## 九、本机操作步骤（自己跑一遍）

只用到本仓库，不依赖 `edurec-platform` / `edurec-engine`。

### 前置：两个 CSV 要放在项目根目录

| 文件 | 是什么 | 必需列 |
| --- | --- | --- |
| `search-实变函数.csv` | `resources` 表导出 | `id`、`title`、`description`、`metadata`、`view_count`、`avg_rating` |
| `user_behaviors.csv` | `user_behaviors` 表导出 | `user_id`、`resource_id`、`action` |

文件名可以改，下面的命令里一并改掉即可。

### 第 1 步：激活环境

```powershell
cd E:\work\learn\EMBEDD-ER
. .\setup_env.ps1
```

`setup_env.ps1` 会把 matplotlib / NLTK / HuggingFace 的缓存目录都指向仓库内的 `env\`。
**必须激活**：本机当前会话对 `%USERPROFILE%\AppData` 不可写，不激活会在这些库首次运行时直接报权限错误。

> ⚠️ **本机 PATH 上的 `python` 不是 venv 里的那个**，实测指向 `C:\Python314\python.exe`，
> 直接敲 `python` 会报 `ModuleNotFoundError: No module named 'numpy'`。
> `setup_env.ps1` 会把 `.venv\Scripts` 插到 PATH 最前面，但为了不依赖这一步，
> 下面的命令**一律用显式路径** `.\.venv\Scripts\python.exe`。
>
> 另外，所有命令都假设**当前目录是项目根**（脚本里用 `r'E:\work\learn\EMBEDD-ER'` 做绝对路径，
> 但两个 CSV 是按相对路径读的）。本机沙箱下进程 CWD 有时会落到父目录，
> 若报 `FileNotFoundError`，先 `cd` 回项目根。
>
> 命令用 PowerShell 的 **单引号 here-string**（`@'` … `'@`）包 Python 代码：它不做变量展开、
> 也不需要转义。**不要改成双引号 here-string（`@"` … `"@`）** —— 那个不认 `\"` 转义，
> 会把反斜杠原样传给 Python 而报 `SyntaxError: unexpected character after line continuation character`。
> 所以 Python 里一律用双引号字符串，别写 `\"`。

### 第 2 步：量一下两个文件的覆盖情况（**最关键的一步**）

推荐质量完全取决于「用户已交互的资源」和「resources 表」的重叠程度。先跑这个：

```powershell
.\.venv\Scripts\python.exe -c @'
import sys; sys.path.insert(0, r"E:\work\learn\EMBEDD-ER")
import warnings; warnings.filterwarnings("ignore")
from recommender.engine import Recommender
r = Recommender.from_platform("search-实变函数.csv", "user_behaviors.csv", verbose=False)
print("资源 %d 条，用户 %d 个" % (len(r.vec_by_raw), len(r.user_courses)))
for uid, items in r.user_courses.items():
    uniq = list(dict.fromkeys(items))
    hit = [c for c in uniq if c in r.vec_by_raw]
    print("  用户 %s: 交互 %d 条 / 去重 %d 个，落在 resources 表里 %d 个 -> %s"
          % (uid, len(items), len(uniq), len(hit), sorted(hit)[:10]))
'@
```

判断标准：

| 每个用户落在 resources 表里的资源数 | 含义 |
| --- | --- |
| **0** | 无法建画像，只能纯热度推荐 |
| **1** | 画像退化成一个点，相似度含义很弱（**当前两个文件就是这个情况**） |
| **≥ 2** | 可以建画像，也才能做留出法评估 |
| **≥ 10** | 推荐结果才开始有参考价值 |

如果数字很小，先回去补数据导出（见第 6 步），否则后面跑出来的排序说明不了问题。

### 第 3 步：跑推荐，看输出

```powershell
.\.venv\Scripts\python.exe -c @'
import sys; sys.path.insert(0, r"E:\work\learn\EMBEDD-ER")
import warnings; warnings.filterwarnings("ignore")
from recommender.engine import Recommender
r = Recommender.from_platform("search-实变函数.csv", "user_behaviors.csv", method="tfidf")
for uid in sorted(r.user_courses):
    r.recommend(uid, top_k=10)      # 打印表格：标题 | 总分 | 相似 | 热度 | 先修 | 说明
'@
```

输出里每一行是：`课程标题 | 总分 | 相似 | 热度 | 先修 | 说明`。
「先修」在平台轨**恒为 0.50（中性）**，因为平台导出的数据里没有概念-课程映射 —— 这是数据现状，不是 bug。

### 第 4 步：换个权重做消融，确认各项真的在起作用

```powershell
.\.venv\Scripts\python.exe -c @'
import sys; sys.path.insert(0, r"E:\work\learn\EMBEDD-ER")
import warnings; warnings.filterwarnings("ignore")
from recommender.engine import Recommender
r = Recommender.from_platform("search-实变函数.csv", "user_behaviors.csv", verbose=False)
cfgs = {"仅相似度": {"sim": 1.0, "hot": 0.0, "prereq": 0.0},
        "仅热度":   {"sim": 0.0, "hot": 1.0, "prereq": 0.0},
        "默认":     {"sim": 0.75, "hot": 0.22, "prereq": 0.03}}
for label, w in cfgs.items():
    print("---", label, "---")
    for uid in sorted(r.user_courses):
        rows = r.recommend(uid, top_k=3, weights=w, explain=False)
        print("  用户", uid, ":", [x["course_id"] for x in rows])
'@
```

如果「仅相似度」和「仅热度」排出完全一样的顺序，说明画像没起作用（通常是第 2 步的覆盖太低，或正文为空）。

### 第 5 步：换嵌入方式对比

```powershell
.\.venv\Scripts\python.exe -c @'
import sys; sys.path.insert(0, r"E:\work\learn\EMBEDD-ER")
import warnings; warnings.filterwarnings("ignore")
from recommender.engine import Recommender
for m in ("tfidf", "doc2vec"):
    r = Recommender.from_platform("search-实变函数.csv", "user_behaviors.csv",
                                  method=m, verbose=False)
    uid = sorted(r.user_courses)[0]
    rows = r.recommend(uid, top_k=5, explain=False)
    print("[%s] 维度=%d" % (m, len(next(iter(r.vec_by_raw.values())))),
          [x["course_id"] for x in rows])
'@
```

* `tfidf`：中文 TF-IDF（jieba 分词 + 2~4 字 n-gram），37 个文档时约 744 维，无需训练，最稳。
* `doc2vec`：项目自带的 Doc2Vec，本入口里**已修正**为真正 `train()`、`vector_size=100`、`epochs=60`。
  语料只有 37 篇时向量质量一般，仅作对照。

> ⚠️ **不要**直接调用 `embeddings/tfidf.py` 和 `embeddings/doc2vec.py` 的原始函数：
> `tfidf_embeddings()` 默认 `reduce=True` 会 TSNE 到 **700 维**，37 个文档必然报错；
> `doc2vec_embeddings()` **建了 vocab 却从不训练**，返回的是随机向量。
> 平台入口已绕开这两个缺陷。

### 第 6 步：如果覆盖太低，怎么补数据

想真正评估，两个文件必须覆盖同一批资源。三选一：

| 做法 | 说明 |
| --- | --- |
| **导出全量 `resources` 表** | 而不是按关键词筛出的那几十条（推荐） |
| **按行为日志的资源 id 过滤导出** | 保证每个用户的已交互资源都在候选池里 |
| **换一批同源数据** | 例如用 `edurec-platform/backend/data/snapshots/<run_id>/` 下的 `resources.csv` + `behaviors.csv`（契约格式，列名不同，见 §2.3） |

> 注意：快照格式（`resources.csv` 是 `resource_id,title,...`）与本入口要求的
> **表导出格式**（`id,title,...`）列名不同，不能直接喂进来。要么改导出列名，要么按 §2.3/§2.4 的差异做一次映射。

### 常见报错对照

| 报错 | 原因 | 修法 |
| --- | --- | --- |
| `PermissionError: ...\.matplotlib\fontlist-...lock` | 没激活 `setup_env.ps1` | 先 `. .\setup_env.ps1` |
| `FileNotFoundError: ...\output\graphs\...` | 进程 CWD 不在项目根 | `cd` 到项目根，或用绝对路径 |
| `KeyError: 'id'` / `'description'` | CSV 不是表导出格式（缺列） | 确认列名，见第 1 步的必需列表 |
| `ValueError: 未知 method: ...` | `method` 只支持 `tfidf` / `doc2vec` | 改参数 |
| `TypeError: object of type 'int' has no len()` | 用了旧版 `engine.py` 的打印逻辑 | 本仓库已修（`_label()` 兼容整数 ID） |
| 相似度全是同一个值 | 图像只有一个资源，或正文全空 | 看第 2 步覆盖；确认 `description` 非空 |

---

# 附录：推荐效果评测（HitRate / NDCG / MRR）

## A1. 评测协议

| 项 | 设定 |
| --- | --- |
| 数据轨道 | MOOCCube |
| 候选集 | 已嵌入的 **582 门**课程（`output/graphs/v01` + `rec_cache/embeddings.npz`） |
| 交互 | `data/MOOCCube/relations/user-course.json`（682,753 条） |
| 用户抽样 | 已学 ≥ 3 门候选课程的用户中随机抽 **2000** 个（seed=0，可复现） |
| 留出 | 每个用户按交互顺序留出**最后 1 门**作为测试目标 |
| 画像 | 剩余已学课程向量的**均值**（原项目做法） |
| 排序 | 对候选池全部课程打分降序；**只排除训练部分的已学课程**，留出项保留在候选内 |
| 指标 | HR@5、HR@10、NDCG@10、MRR@10 |

脚本：
* `eval_original_methods.py` —— 纯语义（各嵌入方法 + 余弦）
* `eval_hybrid_engine.py` —— 混合引擎（`recommender/engine.py`）
* 指标实现：`Recommender.evaluate_topk()`

> ⚠️ **协议局限**：`user-course.json` 只有 `(user, course)` 对，**没有时间戳**，
> 所以"最后 1 门"是**文件顺序而非真实时序**，留出项不代表"下一次会学什么"。
> 这是本评测最大的近似，会影响绝对值（见 A7）。
>
> ⚠️ 原项目 `Classification.ipynb` 用的是**分类**评估（`classify_cv` → accuracy），
> 与本附录的**排序**评估是两种任务，数字**不可直接互比**。

---

## A2. 主结果：纯语义 vs 混合引擎

同一协议、同一 2000 用户：

| 方法 | HR@5 | HR@10 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| 随机向量（下界） | 0.0070 | 0.0120 | 0.0057 | 0.0038 |
| 热门基线（按训练集频次排序） | 0.2320 | 0.3185 | 0.1842 | 0.1429 |
| 纯语义 · EMBEDD-ER | 0.0810 | 0.1340 | 0.0701 | 0.0509 |
| 纯语义 · TF-IDF（原样） | 0.0220 | 0.0395 | 0.0206 | 0.0149 |
| 纯语义 · TF-IDF（不降维） | 0.0275 | 0.0520 | 0.0259 | 0.0182 |
| 纯语义 · Doc2Vec（原实现） | 运行报错（见 A6） | — | — | — |
| 纯语义 · Doc2Vec（修正 train） | 0.1030 | 0.1425 | 0.0846 | 0.0669 |
| 纯语义 · BERT(bert-base-chinese) | 0.1250 | 0.1750 | 0.1071 | 0.0864 |
| **混合引擎（.75/.22/.03）** | **0.2720** | **0.3635** | **0.2096** | **0.1620** |

**结论**：
1. 纯语义的**最好**结果（BERT 0.1750）**仍不及热门基线**（0.3185）—— 与 `edurec-engine` 契约文档记录的
   "语义召回 HitRate@10 = 0.4432 低于热门基线 0.4930" 是同一个现象。
2. 但纯语义相对随机有 **11~15 倍**提升（EMBEDD-ER 11.2×、BERT 14.6×），说明学到的信号是真实的，只是弱。
3. **混合引擎全面超过热门基线**：HR@5 +17.2%、HR@10 +14.1%、NDCG@10 +13.8%、MRR@10 +13.4%。
   相对最好的纯语义（BERT）：HR@5 **2.18 倍**、HR@10 **2.08 倍**、NDCG@10 **1.96 倍**。

---

## A3. 消融：各信号的真实贡献

| 配置 | HR@5 | HR@10 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| 仅 Sim (1,0,0) | 0.0810 | 0.1340 | 0.0701 | 0.0509 |
| 仅 Hotness (0,1,0) | 0.2270 | 0.3160 | 0.1828 | 0.1419 |
| 仅 Prereq (0,0,1) | 0.0075 | 0.0125 | 0.0054 | 0.0032 |
| **默认 (.75,.22,.03)** | **0.2720** | **0.3635** | **0.2096** | **0.1620** |

* **Hotness 是主力**（单独就接近热门基线），Sim 提供增量（0.3160 → 0.3635，+15%）。
* **Prereq 单独几乎无效**（0.0125，接近随机 0.0120）。原因见 `DATA_FLOW.md` §4.2：
  依赖图只覆盖 361 个概念，与课程概念交集仅 355 个，且候选课程的先修需求普遍能被满足
  （"要么大家都满足、要么都没法判定"），因此缺乏区分度。
* 两者**互补而非冗余**：Sim 单独 0.1340 + Hotness 单独 0.3160，合起来 0.3635，高于任一单项。

---

## A4. 权重扫描（γ 固定 0.03）

| sim | hot | HR@5 | HR@10 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0.88 | 0.2225 | 0.3155 | 0.1834 | 0.1427 |
| 0.20 | 0.70 | 0.2415 | 0.3225 | 0.1924 | 0.1519 |
| 0.40 | 0.53 | 0.2600 | 0.3510 | 0.2087 | 0.1649 |
| **0.60** | **0.35** | 0.2680 | **0.3770** | **0.2191** | **0.1706** |
| **0.75（默认）** | 0.22 | **0.2720** | 0.3635 | 0.2096 | 0.1620 |
| 0.90 | 0.09 | 0.2155 | 0.2975 | 0.1736 | 0.1356 |
| 1.00 | 0.00 | 0.0540 | 0.0955 | 0.0468 | 0.0322 |

* 最优点在 **sim≈0.60~0.75**，两端都显著变差。
* `sim=0.60` 在 HR@10 / NDCG@10 / MRR@10 上更好，`sim=0.75` 在 HR@5 上更好 ——
  **默认 0.75 偏保守，调到 0.60 更均衡**（HR@10 +3.7%，NDCG +4.5%）。
* 单靠相似度（sim=1.0）会掉到 0.0955，比随机好但远不如热门基线。

---

## A5. 画像长度

| 画像取法 | HR@5 | HR@10 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| 取前 3 门 | 0.2675 | 0.3620 | 0.2035 | 0.1546 |
| 取前 5 门 | 0.2705 | 0.3615 | 0.2068 | 0.1589 |
| 取前 10 门 | 0.2715 | 0.3615 | 0.2086 | 0.1612 |
| 全部已学 | 0.2720 | 0.3635 | 0.2096 | 0.1620 |

**差异极小**（HR@5 0.2675→0.2720，+1.7%）。用户已学课程中位仅 4 门，所以"取前 3 门"就已经接近全量。
这也说明**均值池化对长历史没有收益** —— 想利用长历史需要加权（如近期优先）或序列模型。

---

## A6. 原项目各嵌入方法的可用性（重要）

评测中实测到的原实现问题：

| 原函数 | 状态 | 原因 |
| --- | --- | --- |
| `embeddings/tfidf.py::tfidf_embeddings` | ⚠️ 可跑但**负优化** | 默认 `reduce=True` 调 `TSNE(n_components=700)`：① 文档数 < 700 时直接报错（`perplexity must be less than n_samples`）；② 582 篇时能跑，但指标从 HR@10 0.0520 **降到 0.0395**（−24%） |
| `embeddings/doc2vec.py::doc2vec_embeddings` | ❌ **原样报错** | `sentence_preprocessing` 用 `nltk.corpus.stopwords`（英文），未激活环境时找不到语料直接 `LookupError` |
| 同上（即便语料可用） | ❌ **逻辑错误** | 函数内**建了 vocab 却从不 `train()`**，返回的是未训练模型上的 `infer_vector`，即**随机向量**；且预处理 `re.sub(r'[^\w\s]','')` 会把中文全部删掉 |
| `embeddings/bert.py::bert_embeddings` | ✅ 可用 | 但硬编码 `model_name="bert-base-uncased"`（英文词表），对中文语料基本无效；本评测通过其已参数化的 `model_name` 传入 `bert-base-chinese` |
| `embeddings/embedder.py::embedder_embeddings` | ✅ 可用 | 即 EMBEDD-ER 本体；本评测直接复用 `rec_cache/embeddings.npz` 中已产出的向量 |

---

## A7. Text 构成的影响（同一方法，只换文本）

方法固定为 TF-IDF（不降维）：

| Text 构成 | HR@5 | HR@10 | NDCG@10 | 中位长度 |
| --- | --- | --- | --- | --- |
| 全文本（`Generation.ipynb` 原样） | 0.0275 | 0.0520 | 0.0259 | 1574 字 |
| 去掉章节编号 | 0.0525 | 0.0645 | 0.0427 | 905 字 |
| **只留标题 + 简介** | **0.1115** | **0.1285** | **0.0978** | 337 字 |

**这是本次评测最有操作价值的发现**：`chapter` 字段是 `01.01.01.01` 这类**章节编码，纯噪声**，
而 `Generation.ipynb` 构建 `Text` 时特意把它拼了进去。去掉后 HR@5 **提升 1.9 倍**，
再进一步只留标题+简介则**提升 4.1 倍**。

---

## A8. 评测实现中的坑（复现时务必注意）

1. **留出项不能被当成"已交互"排除。** 第一版把 `held` 也设成 `-inf`，
   等于把预测目标删掉，导致所有方法（含热门基线）指标**全是 0.0000**。
   正确做法：只排除**训练部分** `rest`。
2. **相似度缓存必须随嵌入失效。** `Recommender._sim_mat` 缓存了归一化后的候选矩阵，
   直接改 `vec_by_raw` 不会重建，换嵌入后三种方法会输出**完全相同的指标**。
   已提供 `set_resource_vectors()` 统一处理。
3. **先修闭包与用户概念集合必须缓存。** 二者跨多组权重配置完全相同，
   不缓存会把评测从秒级拖到**数十分钟直至卡死**（实测 CPU 累计 1400s+、内存 8 GB）。
4. **高维嵌入避免逐用户复制子矩阵。** TF-IDF 达 17,173 维时，
   每次 `_sim_mat[cols]` 会复制约 80 MB，2000 用户即上百 GB 分配量。
   已改为整表矩阵乘法 + fancy indexing。
5. **TF-IDF(17173 维) 的完整 2000 用户评测在本机未跑完**（单机内存与时间成本过高），
   A2 中该行为 582 维 TF-IDF（不降维）的结果，A7 的 17,173 维结果只跑了文本构成对比。

---

## A9. 结论

1. **纯语义（原项目做法）在这套数据上打不过热门基线。** 最好的 BERT 也只有 HR@10 0.1750，
   低于热门基线 0.3185。这是数据特性决定的：用户选课高度集中于少数热门课
   （最热一门被 16.4% 的用户选过），而"用户内选课相似度"仅比随机配对高 **+0.017~+0.038**（弱信号）。
2. **混合引擎显著优于两者。** HR@5 0.2720 / HR@10 0.3635 / NDCG@10 0.2096，
   全面超过热门基线（+14~17%）与最好的纯语义（约 2 倍）。
3. **Hotness 是主力、Sim 提供增量、Prereq 目前无效**（依赖图覆盖太小）。
4. **最该修的是语料构造**：把章节编码从 `Text` 里去掉，同方法下 HR@5 提升近 2 倍、只留标题+简介提升 4 倍。
   这比调权重收益大得多。
5. 默认权重建议从 `(.75,.22,.03)` 调整为 `(.60,.35,.03)`，HR@10 / NDCG / MRR 均更优。
