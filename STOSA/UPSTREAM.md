# STOSA：上游代码 + 本地复现改动

- 上游：<https://github.com/zfan20/STOSA>（WWW'22，DOI 10.1145/3485447.3512077）
- 克隆基线：`1f00ecb Update README.md`（`origin/main`）
- 入库方式：代码与文档直接入库；2026-11-12 整理仓库时移除了内嵌的 `.git`
  （以免变成 embedded git repo），上游历史不再随本仓库携带。

## 我们对上游代码的改动

只有 3 行，全部是 Python / NumPy 版本兼容，**没有改模型结构、损失函数或训练流程**：

| 文件:行 | 改动 | 原因 |
| --- | --- | --- |
| `trainers.py:12` | `from utils import` 去掉 `get_user_performance_perpopularity, get_item_performance_perpopularity` | 上游 `utils.py` 里没有这两个符号，import 阶段就 `ImportError` |
| `utils.py:261` | `np.float` → `np.float64` | NumPy 1.24 起弃用、后续版本移除 `np.float` 别名 |
| `utils.py:355` | `np.asfarray(r)` → `np.asarray(r, dtype=np.float64)` | NumPy 2.0 移除 `np.asfarray` |

## 上游的 bug（未修，只在文档里记录）

- `main.py:125-128`：`--do_eval` 分支没有把 `args.train_matrix` 换成 test 矩阵，
  直接跑会让 NDCG@10 −24%、MRR −34%。
- `main.py:140`：`DistSAModel` 硬编码 `patience=100`，论文里是 50。

细节、复现数字与三条结论见 `REPRODUCTION_REPORT.md`；论文精读见 `STOSA论文阅读笔记.md`。

## 复现命令

```bash
python main.py --data_dir=./data/ --model_name=DistSAModel --data_name=Beauty \
  --output_dir=outputs/ --lr=0.001 --hidden_size=64 --max_seq_length=100 \
  --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 \
  --num_attention_heads=4 --attention_probs_dropout_prob=0.0 \
  --pvn_weight=0.005 --epochs=500
```

## 哪些东西不入库

| 路径 | 体积 | 说明 |
| --- | --- | --- |
| `outputs/`、`outputs_smoke/` | 每个 checkpoint 6.7 MB | 训练产物，跑一次即可重新生成 |
| `*.pdf` | 8.8 MB | 论文 PDF，只保留提取出的 `paper_text.txt` 与阅读笔记 |

`data/Beauty.txt`（1.1 MB，22,363 用户 / 12,101 物品）**入库**：
`src/engine` 里的融合实验（`scripts/run_fusion_probe.py`）默认读它，保证 clone 下来能直接复现。
`STOSA/outputs/` 下的 checkpoint 也被融合实验当作冻结的物品分布来源，缺失时按上面的命令重训一次即可。
