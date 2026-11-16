# 引用与来源

本仓库包含两类来源不同的工作，引用时请区分：

## 1. 推荐模型（`src/engine/`）

语义召回 + 质量信号融合的思路参考 CourseHub：

- CourseHub: 语义课程推荐框架（DOI: `10.1109/ICNWC68145.2026.11518402`）。
- 本仓库的实现与论文的差异（学生侧输入、是否训练、质量权重）见
  [`design.md` §1.2](../design/model-design.md#12-与-coursehub-的差异以及为什么)。

## 2. 知识图谱/概念生成（`src/kg/`）

`src/kg/` 派生自 EMBEDD-ER，根目录 `CITATION.cff` 记录的是该论文：

- Bazouzi, A., Foursov, M., Le Capitaine, H., Miklos, Z.
  “EMBEDD-ER: EMBEDDing Educational Resources Using Linked Open Data”,
  CSEDU 2023. DOI: `10.5220/0012045300003470`.

`src/kg/research/` 是论文原型的研究脚本，不在生产路径上；
生产路径只有 `server.py` / `generate.py` / `recommender/pipeline.py` 等
（见 [`../src/kg/README.md`](../../src/kg/README.md)）。

## 3. 数据集

- MOOCCube：教育域课程/选课/概念数据，需自行下载，不入库。
- MovieLens：已移除（旧的 `movielens` 轨道不再支持）。
