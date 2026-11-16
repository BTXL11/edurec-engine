# recommender/

本包同时包含两类代码，边界如下：

- **生产路径**：`pipeline.py` —— `kg_server.py` 的「标题/描述 → 概念」流水线。
- **研究原型**：`engine.py`、`popularity.py`、`prerequisites.py`、`profile.py`、
  `build_embeddings.py`、`idmap.py`、`stats.py` —— EMBEDD-ER 的混合排序实验，
  只被 `src/kg/research/` 引用。

原型的完整说明（数据来源、公式、评测数字）已移到
[`../research/recommender-prototype.md`](../research/recommender-prototype.md)。
生产链路不需要安装这些实验依赖；`pipeline.py` 对嵌入方法是懒加载的。
