# src/kg —— 知识图谱与概念生成（EMBEDD-ER 派生物）

`src/kg` 是一个**真正的 Python 包 `kg`**（与 `src/engine` 对等）：

- 内部导入统一写 `from kg.recommender...` / `from kg.embeddings...` /
  `from kg.preprocessing...`；
- 入口脚本只把 `src` 加入 `sys.path`，不再把 `src/kg` 当源码根；
- 不再依赖 `sys.path.insert` 的副作用来解析内部模块。

本目录只保留**生产路径**上会被用到的代码：

```
server.py             知识生成 HTTP 服务实现（scripts/kg_server.py 调用）
generate.py           标题/描述 -> 概念图的命令行入口
recommender/pipeline.py  资源 -> RDF 图 -> 嵌入 -> 概念（kg_server 只用这一条）
preprocessing/text.py    Wikifier 调用 + RDF 构造
embeddings/            文本/图嵌入方法（pipeline 按需懒加载）
textwiki.py, textlang.py  Wikifier / 文本语言辅助
```

`recommender/` 里其余模块（`engine.py`、`popularity.py`、`prerequisites.py` 等）
是 EMBEDD-ER 原型的研究实现，只被 `research/` 引用，**不在生产路径上**。

## 依赖

知识生成服务的最小依赖是 `pip install -e .[kg]`（`rdflib` + `scikit-learn`）。
`research/` 里的脚本还需要 `gensim` / `wikipedia2vec` / `nltk` / `matplotlib`
等研究依赖，未列入默认安装。

## research/

一次性分析、评测、可视化与环境脚本都收在 [`research/`](research/README.md)，
不在运行时路径上，也不被 `edurec-engine` 的推荐链路导入。它们保留是为了
可复现论文/答辩里的实验数字，不是产品代码。

## 与 `src/engine/` 的关系

- `src/engine/`：平台推荐引擎（语义双塔 + 质量融合），入口 `scripts/train_semantic.py` /
  `scripts/run_batch_infer.py`。
- `src/kg/`：知识图谱/概念生成，入口 `scripts/kg_server.py` 与 `generate.py`。

两者不互相 import；唯一的交汇点是平台侧：推荐结果走文件契约，概念生成走可选的
HTTP 契约（见 `docs/contracts/platform-contract.md`）。
