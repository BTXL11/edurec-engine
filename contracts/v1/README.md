# edurec 交接契约 v1（可执行版本）

本目录把 [`docs/contracts/platform-contract.md`](../../docs/contracts/platform-contract.md) 里的**规范文字**变成
**可校验的机器可读产物**。规范文档仍是唯一权威；这里的东西只做两件事：

1. 用 JSON Schema 固定「快照元信息 / 推荐主文件 / 旁挂信封」的形状；
2. 用 golden fixture + 校验器让两侧测试跑同一份样本，避免文档和实现各说各话。

```
contracts/v1/
  README.md                 本文件
  snapshot.schema.json      meta.json 的 JSON Schema
  snapshot.columns.json     CSV 列定义（必需/可选、类型、说明）
  recommendations.schema.json   推荐主文件的 JSON Schema
  envelope.schema.json      推荐旁挂信封的 JSON Schema
  validate.py               校验器（stdlib，无第三方依赖）
  make_fixtures.py          重新生成 fixtures（确定性）
  fixtures/
    snapshot/               一份最小的合法平台快照
    recommendations.json    一份合法的推荐主文件
    recommendations.meta.json  对应的旁挂信封
    invalid/                故意违规的样本 + cases.json 清单
```

## 版本与所有权

- **契约版本**：`contract_version = 1`。快照、推荐信封各自带版本号。
- **所有权**：engine 维护本目录（schema / fixture / 校验器）；platform 只消费，
  不修改，也不新增字段。任何破坏性变更必须先升版本，再两侧同时改。
- **兼容规则**：
  - 新增**可选**字段/列 = 向后兼容，不升版本；
  - 删除字段、改类型、改键语义、改顺序语义 = 破坏性变更，必须升版本；
  - 平台读到不认识的 `contract_version` 必须明确报错退出（engine 侧已实现）。

## 校验器

在 engine 仓库根目录运行：

```bash
# 校验一份快照目录（meta.json + 5 个 CSV + sha256）
python contracts/v1/validate.py snapshot dataset/platform_snapshot/<run_id>

# 校验 engine 产物（主文件 + 可选信封）
python contracts/v1/validate.py recommendations model/recommendations.json

# 一次校验两侧交接物
python contracts/v1/validate.py all \
  --snapshot dataset/platform_snapshot/<run_id> \
  --recommendations model/recommendations.json
```

退出码：`0` 合法，`1` 违规（打印首个错误），`2` 用法错误。

校验器也被测试直接 import：`tests/test_contract.py` 会对 fixtures 做正反两向断言，
并对 `write_handoff` 的实际输出跑一遍 `recommendations` 校验。

## platform 侧如何复用

platform 的导入服务只认**主文件**，不引 engine 代码。为保证两侧不漂移：

- platform 仓库维护一份**逐字节相同**的 golden fixture：
  `backend/internal/service/testdata/recommendations.contract.json`；
- `scripts/check-contract-sync.sh`（platform 仓库根目录）比对它与本目录的
  `fixtures/recommendations.json`，不一致就失败；
- platform 的 `recommendation_import_contract_test.go` 用这份 fixture 跑导入，
  断言「合法样本能导入、非法样本被拒绝」。

golden fixture 的**唯一来源是本目录**。改契约时先改这里、跑 `make_fixtures.py`，
再在 platform 执行 `bash scripts/check-contract-sync.sh --update` 同步副本。

## 修改契约的流程

1. 改 `docs/contracts/platform-contract.md`（规范）与本目录 schema；
2. `python contracts/v1/make_fixtures.py` 重新生成 fixtures；
3. `python -m pytest tests/test_contract.py` 通过；
4. 在 platform 同步 fixture 并跑 `go test ./internal/service/...`；
5. 若属破坏性变更，提升 `contract_version` 并同步改两侧版本校验。
