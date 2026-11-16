# research/ —— EMBEDD-ER 研究脚本（非生产）

这些脚本来自 EMBEDD-ER 论文原型，用于复现实验、可视化与一次性环境排查。
**它们不在 `edurec-engine` 的运行时路径上**，也不被 `scripts/` 或 `src/engine/` 导入。

| 文件 | 用途 |
| --- | --- |
| `ablation.py`, `compare_scoring.py` | 权重/打分消融 |
| `eval_hybrid_engine.py`, `eval_original_methods.py` | 离线评测（HR/NDCG/MRR） |
| `run_batch.py`, `run_batch_B.py` | 旧版批量构图/嵌入入口 |
| `visualize_kg.py`, `visualize_kg_b.py`, `plotting/` | 知识图谱可视化 |
| `check_batch.py`, `check_concept_noise.py`, `check_kernel.py` | 一次性数据质量检查 |
| `classification/`, `similarity/` | 原论文的分类/相似度工具 |
| `convert_w2v.py`, `fetch_*.py`, `find_hub_version.py`, `smoke_bert_gpu.py`, `verify_env.py` | 本机环境/权重准备脚本 |

运行方式（在 engine 仓库根目录）：

```bash
.venv/bin/python src/kg/research/eval_hybrid_engine.py
```

脚本会把 `src` 与自身目录加入 `sys.path`，因此能 import
`kg.recommender.*` / `kg.embeddings.*` / `kg.preprocessing.*`（生产包）。
旧的 `recommender` 原型说明见 [`recommender-prototype.md`](recommender-prototype.md)。
