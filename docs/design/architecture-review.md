# edurec-engine 架构评估与重构记录

> 评估日期：2026-11-12
> 范围：`src/engine`（推荐主链路）、`src/kg`（可选知识服务）、`scripts`、`contracts`
> 本轮性质：**评估 + 设计 + 已实施的重构**（未实施项明确标为路线图）

---

## 0. 结论摘要

engine 的总体设计是清晰且可复现的：单向分层（`data → features → models → pipeline`）、
固定随机种子、契约可执行、评估协议有对照基线。问题不在「方向」，而在**实现细节的规模化与
口径一致性**：

1. **数据访问没有索引**：按用户扫行为、按资源扫资源，全量推理时是 O(U×B) / O(R²)。
2. **质量融合被重复归一化**：默认权重和恰好为 1，掩盖了错误；换权重即失真。
3. **编码器缓存不随语料失效**：缓存键只有文本 hash，换语料后同一文本命中旧 IDF 下的向量。
4. **配置没有集中校验**：非法比例/权重/枚举要到计算中途才报错。
5. **高斯塔 checkpoint 无法被批量推理消费**：脚本硬编码确定性塔，属于「能训练、不能交付」。

本轮修复 1–5（见 §4），并新增回归测试。重构后 engine 测试从 168 增至 181，全绿。

---

## 1. 基线

```bash
cd edurec-engine
.venv/bin/python -m pytest -q
# 168 passed（重构前）
# 181 passed（重构后）
```

评估方法：从数据流入口（`data/*.py`）沿调用链读到交付出口（`infer_batch.write_handoff`
→ `contracts/v1/validate.py`），重点检查：

- 同一口径（资源文本、需求文本、语料指纹）是否只有一处实现；
- 复杂度是否随「用户数 × 行为数」放大；
- 失败是否明确（而不是静默降级为错误结果）；
- 默认配置之外的参数组合是否会暴露逻辑错误。

---

## 2. 目标架构（设计）

### 2.1 分层

```
contracts/v1       交接契约（schema + fixtures + 校验器），与模型无关
      ▲
scripts            训练 / 推理 / 服务入口（只做参数装配与编排）
      ▲
pipeline           批量推理、评估协议、交接文件写出
      ▲
models/recall      双塔模型、训练循环、打分
      ▲
features           文本编码器（可插拔 + 缓存）
      ▲
data               DataBundle、各数据源加载器、BundleAccess（索引化读模型）
```

**规则**：下层不 import 上层；文本构造这类「数据口径」放在最底层 `data/schema.py`，
避免 `data → features` 的反向依赖（本轮已修正）。

### 2.2 读模型显式化

`DataBundle` 是事实表；`BundleAccess` 是它的索引化只读视图：

```
DataBundle ──► BundleAccess
                 ├─ resources_by_id / text_by_id
                 ├─ behaviors_by_user（按时间序）
                 ├─ seen_by_user
                 ├─ popularity
                 └─ ratings_by_item
```

训练、评估、推理都从同一个视图取数，避免三套「怎么找用户历史 / 怎么拼文本」的实现漂移。

### 2.3 失败要响

- `EngineConfig.validate()`：入口集中校验，非法组合在加载数据前失败。
- checkpoint 校验：`encoder_version` + `corpus_fingerprint` 双校验；不一致直接退出。
- 不支持的 `tower_kind`：批量推理明确拒绝，不静默按错误结构加载。

---

## 3. 发现的问题

| # | 位置 | 问题 | 影响 | 本轮处置 |
|---|---|---|---|---|
| E1 | `pipeline/recall_pipeline.py`、`pipeline/infer_batch.py` | 多处 `[next((resource_text(r) for r in bundle.resources if r.resource_id == i), "") for i in item_ids]`；按用户 `[b for b in bundle.behaviors if b.user_id == u]` | O(R²) / O(U×B)；10 万行为、5000 用户时单次全量扫描约 **9 秒** | ✅ 新增 `data/access.py::BundleAccess` 并替换 |
| E2 | `pipeline/infer_batch.py` | 先 `fuse_quality(semantic=0)`（内部已除权重和），逐用户融合时再除一次 | 权重和 ≠ 1 时质量信号被平方级压低；默认 0.6/0/0.4 恰好掩盖 | ✅ 统一走 `_fuse_user_scores`，只除一次；新增回归测试 |
| E3 | `features/encoder.py::CachedEncoder` | 缓存键只有文本 hash；缓存路径由 fit **之前**的 version 生成；`fit` 后不更新 version | 换语料后同一文本命中旧 IDF 向量，训练/推理口径被污染 | ✅ 缓存写入 `__version__` / `__corpus__` 元数据，加载时校验；`fit` 同步 version |
| E4 | `config.py` | 没有集中校验 | 非法比例/权重/枚举在训练中途或评估空集才暴露 | ✅ `EngineConfig.validate()`，两个 CLI 入口调用 |
| E5 | `scripts/run_batch_infer.py` | 硬编码 `TwoTowerSemantic`；高斯塔学生塔输入多一维，state_dict 形状不同 | `--tower-kind gaussian` 训练出的 checkpoint 无法推理（报错或静默错误） | ✅ 检测 `ckpt["tower_kind"]`，非 deterministic 明确退出 |
| E6 | `features/encoder.py` | 类名 `LocalTfidfSvdEncoder` 与实际实现不符（SVD 已移除，只剩 hash + IDF） | 阅读者按名字推断行为会误解 | ⏳ 路线图 P2（改名为 `LocalHashIdfEncoder`，保留别名） |
| E7 | `config.py` | `EngineConfig` 是 50+ 字段的 god dataclass | 参数归属不清、测试难构造、YAML 演进困难 | ⏳ 路线图 P2（拆成 Data/Encoder/Recall/Quality 四组，保留 facade） |
| E8 | `models/recall/gaussian.py` | 已实现、已评测、默认不启用，但仍留在生产包路径 | 维护成本与误用风险（见 E5） | ⏳ 路线图 P3（移入 `research/` 或加显式 enable 开关） |
| E9 | `features/encoder.py` | 需求文本构造在 `encoder.user_need_text`、`BundleAccess.need_text`、`recall_pipeline.need_text_from` 三处相似实现 | 口径容易漂移 | ⏳ 路线图 P2（以 `BundleAccess` 为唯一实现，其余转调） |

### 3.1 复杂度实测

在 `R=3000` 资源、`U=5000` 用户、`B=100000` 行为上：

```
items R=3000:       旧 O(R^2) = 60.1ms，BundleAccess(build+texts) = 53.9ms
histories U=5000:   线性扫描 = 9013.7ms，索引访问 = 1.046ms   （约 8600×）
```

资源文本在 3000 规模下差距不大（`next` 平均命中前半段），但历史查找是数量级差异；
这正是全量推理与逐用户需求构造的主循环。

---

## 4. 本轮重构落地

### 4.1 `data/access.py`：索引化只读视图（E1）

新增 `BundleAccess`，提供：

- `item_ids` / `text_by_id` / `item_texts(ids)`；
- `history(user)` / `seen(user)` / `need_text(user, max_items)` / `need_texts(users)`；
- `popularity` / `ratings_by_item` / `mean_rating(rid)`。

替换点：

- `pipeline/infer_batch.py`：文本、热度、评分、item_ids 全部从 `BundleAccess` 取；
- `pipeline/recall_pipeline.py`：四处 O(R²) 的 `next(...)` 改为 `access.item_texts(item_ids)`；
  顺带修掉 `popularity_baseline` 里 `item_ids.index(i)` 的 O(n²) 与循环内重复 `set(order)`；
- `models/recall/trainer.py::build_interactions`：文本查找改走 `BundleAccess`。

### 4.2 文本构造下沉到数据层（依赖方向修正）

`resource_text` 从 `features/encoder.py` 移到 `data/schema.py`，`encoder.py` 重新导出以保持
既有 import 兼容。这样 `data/access.py` 不再需要反向 import features，分层规则恢复一致。

### 4.3 质量融合只归一化一次（E2）

`infer_batch` 新增纯函数：

```python
def _fuse_user_scores(semantic_norm, rating_norm, popularity_norm, config):
    return fuse_quality(semantic_norm, rating_norm, popularity_norm,
                        w_semantic=..., w_rating=..., w_popularity=...)
```

`rating_norm` / `popularity_norm` 在循环外算一次；每个用户只做加权求和。
新增 `test_fuse_user_scores_divides_weights_once`：权重 (2, 0, 1) 时输出必须是 (2/3, 1/3)。

> 副作用（预期内）：冷启动用户的信封分数从「原始热度计数」改为「秩百分位」，与活跃用户的
> 融合分统一到 [0,1]，便于前端/评估直接比较。

### 4.4 编码器缓存随语料失效（E3）

`CachedEncoder` 现在：

- 写入 npz 时附带 `__version__`（含语料指纹）与 `__corpus__`；
- 加载时比较「去掉 corpus 的版本前缀」：
  - 前缀不同（实现/超参变了）→ 丢弃；
  - 当前 inner 已 fit 且语料指纹不同 → 丢弃；
  - 当前 inner 未 fit（无指纹可用）→ 沿用缓存，保留原有离线复用能力；
- `fit()` 后同步 `self.version` 与 `corpus_fingerprint`，使 checkpoint 的 `encoder_version`
  真正能反映训练语料。

新增 `test_cached_encoder_invalidates_when_corpus_changes`：
同一路径、同一实现、换语料后，编码结果必须等于「新语料下重新 fit 的新编码器」。

### 4.5 配置集中校验（E4）

`EngineConfig.validate()` 覆盖：数据源与快照路径、top_n、训练/验证比例、质量权重非负且和为正、
encoder/tower/sample 枚举、epochs/batch、评估 K 与负样本数。
`scripts/train_semantic.py` 与 `scripts/run_batch_infer.py` 在参数覆盖后调用。

### 4.6 高斯塔推理 fail-fast（E5）

`run_batch_infer.load_model` 读取 `ckpt["tower_kind"]`，非 `deterministic` 时明确退出并给出
可操作信息（重新训练或补齐高斯塔批量入口），避免加载形状不匹配的 state_dict 或静默错误打分。

---

## 5. 后续路线图

### P1（下一轮优先）

1. **统一需求文本实现**：`BundleAccess.need_text` 成为唯一实现；`encoder.user_need_text`
   与 `recall_pipeline.need_text_from` 转调它，消除 E9。
2. **补齐高斯塔批量入口**（若决定继续保留研究能力）：`infer_batch` 需要把 `history_len`
   传给 `score_users`，并让 `load_model` 能构造 `GaussianTwoTower`；否则考虑把高斯模块
   移出生产包路径（E8）。

### P2（结构清晰度）

3. **配置分组**：`DataConfig` / `EncoderConfig` / `RecallConfig` / `QualityConfig`，
   `EngineConfig` 保留为组合 facade，提供属性转发；YAML 按分组写。
4. **编码器改名**：`LocalTfidfSvdEncoder → LocalHashIdfEncoder`，旧名保留别名一个版本周期。
5. **模型注册表**：用 `tower_kind → builder/loader` 的注册表替代 `if kind == ...`，
   让训练与推理共用同一套构造逻辑。

### P3（能力扩展）

6. **多路召回 + 显式去重**：语义 + 热门 + 协同/图；出口统一去重（契约要求已单路满足，
   多路必须补）。
7. **ANN**：资源量增长后把全量点积换成 faiss，`ItemIndex` 接口不变。
8. **序列建模**：若要让不确定性（E8）真正生效，需要行为序列输入，而不是拼接文本。

---

## 6. 验证

```bash
cd edurec-engine
.venv/bin/python -m pytest -q
```

本轮结果：

| 检查 | 结果 |
|---|---|
| `pytest` | **181 passed**（新增 access / cache / fusion / config 回归） |
| 契约校验 | 未改主文件形状，`contracts/v1` 无需变更 |
| 公开 CLI | `train_semantic` / `run_batch_infer` 增加 `cfg.validate()`；参数语义不变 |
