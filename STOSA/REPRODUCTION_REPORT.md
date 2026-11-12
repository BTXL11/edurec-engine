# STOSA 复现报告

> 论文：**Sequential Recommendation via Stochastic Self-Attention** (STOSA)
> 会议：WWW '22（April 25–29, 2022, Virtual Event, Lyon, France）
> 作者：Ziwei Fan\*, Zhiwei Liu, Alice Wang, Zahra Nazari, Lei Zheng, Hao Peng, Philip S. Yu
> DOI: [10.1145/3485447.3512077](https://doi.org/10.1145/3485447.3512077) · arXiv: [2201.06035v2](https://arxiv.org/abs/2201.06035)
> 代码：<https://github.com/zfan20/STOSA>
> 复现时间：2026-11-09 · 数据集：Amazon **Beauty** · 轮次：183 epoch（早停）

---

## 0. 结论速览

| 项目 | 结果 |
|---|---|
| 能否跑通 | ✅ 端到端跑通（数据 → 训练 → 早停 → 全排序测试） |
| 复现指标 / 论文指标 | **约 90%**（MRR 0.0325 vs 0.0360，差 −9.7%） |
| 与最强 baseline 对比 | 仍高于 SASRec（MRR +11.8%）、与 DT4SR 基本持平（+0.7%），**未达到论文的 +11.54% 提升** |
| 主要偏差来源 | 只跑了 README 的单一配置，未做论文附 C 的超参网格搜索（尤其 `pvn_weight` λ）；早停 patience 与论文不一致 |
| 代码级发现 | `--do_eval` 分支的候选集掩码矩阵用错（`main.py:125-128`），会让 test 指标系统性偏低（已验证） |

**一句话**：模型逻辑复现无误、结果趋势正确，但单次配置只到论文 DT4SR 的水平，距 STOSA 论文值还有约 10% 的差距，需要超参搜索才能追上。

---

## 1. 论文要点

### 1.1 三个被指出的问题

1. **确定性建模丢失不确定性**：SASRec / BERT4Rec 把 item 嵌成固定向量，假设用户兴趣是确定性的；而真实序列里 item 转移常常"不合逻辑"，兴趣波动大的用户本该更"不确定"。
2. **点积注意力无法表达协同传递性（collaborative transitivity）**：点积不满足三角不等式。论文给的例子：$i_x=[0,2], i_y=[1,1], i_z=[2,0]$，$\langle i_x,i_y\rangle=\langle i_y,i_z\rangle=2$，但 $\langle i_x,i_z\rangle=0$ —— 未观测到的转移对就断了。距离度量则天然满足 $d(i_x,i_z)\le d(i_x,i_y)+d(i_y,i_z)$，能把协同相似性"传递"到冷启 item。
3. **BPR 对正负样本之间没有距离约束**：只比较"用户对正样本 vs 负样本"的打分差，不保证正样本在隐空间里真的远离负样本。

### 1.2 三个模块

| 模块 | 论文位置 | 内容 |
|---|---|---|
| 随机嵌入 | §4.1, Eq.(3) | item 用对角高斯 $\mathcal{N}(\mu_s,\Sigma_s)$ 表示；均值表 $M^\mu$ 与协方差表 $M^\Sigma$ 各配一套位置嵌入 $P^\mu, P^\Sigma$ |
| Wasserstein 自注意力 | §4.2, Eq.(5)(6)(7) | 注意力权重取**负 2-Wasserstein 距离** $A_{kt}=-W_2(s_k,s_t)$；均值按 $\tilde A$ 加权求和，方差按 $\tilde A^2$ 加权求和（高斯线性组合性质） |
| ℓ_pvn 正则 | §4.5, Eq.(11)(12) | $\ell_{pvn}=[d_{s_t,j^+}-d_{j^+,j^-}]_+$，与 BPR 相加：$L=\sum-\log\sigma(d_{s_t,j^-}-d_{s_t,j^+})+\lambda\ell_{pvn}+\beta\lVert\Theta\rVert_2^2$ |

关键的正定性手段：协方差处处经过 $\mathrm{ELU}(\cdot)+1$（ELU 值域 $[-1,+\infty)$，加 1 后恒正），见 Eq.(4)(9)。

预测层 Eq.(10)：分数即 $W_2$ 距离，**越小越相关，按升序**出 top-N。

复杂度：$O(nd+n^2d+4n^2+nd^2/2)$，与 SASRec 的 $O(n^2d+nd^2)$ 同阶（附录 A）。

### 1.3 论文报告的 Beauty 结果（Table 1，test）

| 指标 | BPRMF | LightGCN | CML | SML | TransRec | Caser | SASRec | DT4SR | BERT4Rec | **STOSA** |
|---|---|---|---|---|---|---|---|---|---|---|
| Recall@1 | 0.0082 | 0.0064 | 0.0072 | 0.0069 | 0.0085 | 0.0112 | 0.0129 | 0.0143 | 0.0119 | **0.0193** |
| Recall@5 | 0.0300 | 0.0287 | 0.0249 | 0.0279 | 0.0321 | 0.0309 | 0.0416 | 0.0449 | 0.0396 | **0.0504** |
| NDCG@5 | 0.0189 | 0.0174 | 0.0184 | 0.0173 | 0.0204 | 0.0214 | 0.0274 | 0.0296 | 0.0257 | **0.0351** |
| MRR | 0.0216 | 0.0203 | 0.0198 | 0.0191 | 0.0236 | 0.0231 | 0.0291 | 0.0323 | 0.0294 | **0.0360** |

---

## 2. 论文方法 → 代码映射

| 论文 | 实现位置 | 说明 |
|---|---|---|
| Eq.(5) $W_2^2=\lVert\mu_1-\mu_2\rVert^2+\mathrm{tr}(\Sigma_1+\Sigma_2-2(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2})$ | `modules.py:22` `wasserstein_distance`、`modules.py:30` `wasserstein_distance_matmul` | 对角协方差下化简为 $\sum(\sqrt{\sigma_1}-\sqrt{\sigma_2})^2$，代码里正是 `(cov1_sqrt-cov2_sqrt)**2` 求和；矩阵版用 `mean1_2+mean2_2-2·mm` 实现，可批量算 |
| 负距离作注意力权重 | `modules.py:247` `attention_scores = -wasserstein_distance_matmul(...)` | `--distance_metric=wasserstein` 时走这条；另有 `kl` 分支（`modules.py:249`） |
| 协方差正定：ELU+1 | `seqmodels.py:112` `sequence_emb = elu_act(self.dropout(sequence_emb)) + 1`；`trainers.py:463` `test_item_cov_emb = elu_activation(...) + 1` | 与 Eq.(4)(9) 一致 |
| BPR + ℓ_pvn | `trainers.py:380`（BPR）、`trainers.py:382`（ℓ_pvn） | `pvn_loss = pvn_weight * sum(clamp(pos_logits - pos_vs_neg, 0) * istarget) / sum(istarget)` |
| Eq.(10) 距离越小越靠前 | `trainers.py:588-596` | 已交互 item 的距离置 `1e+24` 屏蔽，然后 `argpartition` 取前 40 + 升序 `argsort` |
| 全排序（非采样评估） | `trainers.py:483-504` `dist_predict_full` 分块打分 | 对全部 12,103 个候选打分，显存受限时按 `eval_batch_size` 分块 |

结论：**模型层与论文公式逐条对应，未见实现性偏离。**

---

## 3. 复现环境

| 项 | 值 |
|---|---|
| OS | Windows |
| Python | 3.14.0 |
| PyTorch | 2.11.0+cu126（CUDA 12.6） |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB（驱动 616.64） |
| NumPy / SciPy / tqdm | 2.4.1 / 1.17.0 / 4.65.0 |
| 论文环境 | PyTorch + RTX 3090, 64 GB 内存 |

> 差别：论文用 3090，本次用 8 GB 笔记本卡，因此 `batch_size` 保持默认 256、评估分块走 `dist_predict_full` 的分批路径。

---

## 4. 数据集

Amazon Beauty，leave-one-out 划分（最后一交互做 test、倒数第二做 valid，5-core 过滤）。实测统计与论文 Table 2 对比：

| 项 | 论文 Table 2 | 实测 | 是否一致 |
|---|---|---|---|
| #users | 22,363 | 22,363（行数） | ✅ |
| #items | 12,101 | 12,101（`max_item`） | ✅ |
| #interactions | 198,502 | 198,502（逐行 token 数 − 用户数） | ✅ |
| avg. interactions / user | 8.3 | 8.88（198,502 / 22,363） | ⚠️ 论文派生列与其自身原始数字不符 |
| density | 0.05% | 0.073%（198,502 / (22,363×12,101)） | ⚠️ 同上 |

三项原始计数**完全一致**，说明数据版本与预处理没问题；后两列是论文自己的四舍五入/笔误，不影响复现。

---

## 5. 为跑通所做的代码修改

原仓库面向 Python 3.7 + NumPy 1.x，在本机（Python 3.14 + NumPy 2.x）有三处硬阻塞，已做最小改动：

| 文件:行 | 原代码 | 改为 | 原因 |
|---|---|---|---|
| `trainers.py:12` | `from utils import recall_at_k, ndcg_k, get_metric, cal_mrr, get_user_performance_perpopularity, get_item_performance_perpopularity` | 去掉后两个函数 | `utils.py` 中不存在这两个函数，且代码中从未调用 → ImportError |
| `utils.py:261` | `np.reciprocal(np.where(r==1)[0]+1, dtype=np.float)` | `dtype=np.float64` | NumPy 2 移除 `np.float` 别名 |
| `utils.py:355` | `r = np.asfarray(r)[:k]` | `r = np.asarray(r, dtype=np.float64)[:k]` | NumPy 2 移除 `np.asfarray` |

三处均不改变算法语义，只做 API 兼容。**未改动任何模型/损失/评估逻辑。**

---

## 6. 运行配置 vs 论文搜索空间

| 超参 | 论文（附 C：网格搜索范围） | 本次复现 | 备注 |
|---|---|---|---|
| embedding dim | {32, 64}（STOSA 因双嵌入减半） | 64 | |
| max_seq_length | {50, 100} | 100 | |
| num_layers | {1, 2, 3} | 1 | |
| num_heads | {1, 2, 4} | 4 | |
| dropout | {0.3, 0.5, 0.7} | 0.3（attention dropout 0.0） | |
| lr | {1e-3, 1e-4} | 1e-3 | |
| weight_decay | {1e-1, 1e-2, 1e-3} | 0.0 | 论文没给 0，属偏置 |
| **pvn_weight λ** | {0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1 … 1.0}，逐数据集调 | **0.005**（README 默认值） | **最可疑的偏差源**，见 §9 |
| batch_size | 未给 | 256（默认） | |
| epochs / 早停 | 验证 MRR 连续 50 epoch 不升则停 | 上限 500，`main.py:140` 对 DistSAModel 用 **patience=100** | 与论文的 50 不一致 |
| seed | 未给 | 42（`main.py:47`，已 `set_seed`） | 单次运行 |

命令（README 原样，只补了 `--data_dir/--output_dir`）：

```bash
python main.py --data_dir=./data/ --model_name=DistSAModel --data_name=Beauty \
  --output_dir=outputs/ --lr=0.001 --hidden_size=64 --max_seq_length=100 \
  --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 \
  --num_attention_heads=4 --attention_probs_dropout_prob=0.0 \
  --pvn_weight=0.005 --epochs=500
```

模型规模：**1,684,315** 参数。训练耗时：183 epoch 共 **约 64 分钟**（≈21 s/epoch），起止 20:34 → 21:38。

---

## 7. 结果

### 7.1 主结果（Beauty，test，全排序）

论文 STOSA 用 Recall@{1,5} / NDCG@5 / MRR；本仓库输出 HIT@{1,5,10,20,40} / NDCG@{5,10} / MRR，其中 **HIT@N 即 Recall@N**（同一实现 `recall_at_k`），可直接对齐：

| 指标 | 论文 STOSA | 本次复现 | 相对差距 |
|---|---|---|---|
| Recall@1 | 0.0193 | **0.01637** | −15.2% |
| Recall@5 | 0.0504 | **0.04539** | −10.0% |
| NDCG@5 | 0.0351 | **0.03142** | −10.5% |
| MRR | 0.0360 | **0.03252** | −9.7% |

补充指标（同样在 test 上、同一最优权重）：

| HIT@10 | HIT@20 | HIT@40 | NDCG@10 |
|---|---|---|---|
| 0.06596 | 0.09162 | 0.12722 | 0.03803 |

**放在 baseline 里看**（论文 Table 1 的 Beauty 行）：

| 模型 | Recall@1 | Recall@5 | NDCG@5 | MRR |
|---|---|---|---|---|
| SASRec | 0.0129 | 0.0416 | 0.0274 | 0.0291 |
| DT4SR | 0.0143 | 0.0449 | 0.0296 | 0.0323 |
| 论文 STOSA | 0.0193 | 0.0504 | 0.0351 | 0.0360 |
| **本次复现** | 0.01637 | 0.04539 | 0.03142 | 0.03252 |
| 复现 vs SASRec | **+26.9%** | **+9.1%** | **+14.7%** | **+11.8%** |
| 复现 vs DT4SR | +14.5% | +1.1% | +6.1% | +0.7% |

即：**方向对了（明显优于 SASRec），量级只到论文 DT4SR 的位置**。

### 7.2 训练过程

| epoch | train loss | pvn loss | train AUC | HIT@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|---|
| 0 | 0.8597 | 0.2103 | 0.6151 | 0.02656 | 0.01288 | 0.01020 |
| 5 | 0.4458 | 0.0857 | 0.8763 | 0.04758 | 0.02591 | 0.02197 |
| 10 | 0.3315 | 0.0840 | 0.9344 | 0.06685 | 0.03873 | 0.03363 |
| 20 | 0.2201 | 0.0672 | 0.9679 | 0.07539 | 0.04301 | 0.03684 |
| 40 | 0.1313 | 0.0404 | 0.9819 | 0.08018 | 0.04613 | 0.03941 |
| 60 | 0.0885 | 0.0241 | 0.9868 | 0.08237 | 0.04823 | 0.04163 |
| **82（最优）** | 0.0594 | 0.0117 | 0.9893 | **0.08317** | **0.04919** | **0.04250** |
| 100 | 0.0447 | 0.0063 | 0.9906 | 0.08358 | 0.04906 | 0.04236 |
| 140 | 0.0292 | 0.0030 | 0.9928 | 0.08206 | 0.04720 | 0.04035 |
| 182 | 0.0228 | 0.0019 | 0.9936 | 0.08380 | 0.04805 | 0.04085 |

读出三件事：

1. **指标在第 10–60 epoch 就把 90% 的收益吃完了**（MRR 0.0102 → 0.0416），之后是缓慢微调。
2. **训练 loss 一路降到 0.0228、AUC 升到 0.9936，验证指标却在 82 之后走平甚至回落** —— 典型过拟合，这也是早停在 182 触发的原因。
3. **ℓ_pvn 有效收敛**：pvn loss 从 0.21 降到 0.0019，说明正负样本距离约束在后期已基本满足（论文 Eq.(12) 所说的"理想情形第二项趋 0"）。

验证集最优（select 用）：HIT@1 0.02307 / HIT@5 0.05912 / NDCG@5 0.04145 / MRR 0.04250 —— 比 test 高 25–30%。这不是过拟合泄漏，而是 leave-one-out 下 valid 任务本身更"好猜"（valid 目标是倒数第二个 item，且它的历史更短、候选干扰更少），论文同样只报 test，故以 test 为准。

### 7.3 可复现性验证（checkpoint 回放）

用保存的最优权重（`outputs/…-0.005.pt`）单独做一次测试，**逐位复现**了训练结束时的 test 结果：

```
HIT@1 0.01636632  HIT@5 0.04538747  HIT@10 0.06595716
NDCG@5 0.03141800 NDCG@10 0.03803270 MRR 0.03252144
```

→ 权重文件可用、结果不是一次性偶然。

---

## 8. 复现过程中发现的问题

### 8.1 `--do_eval` 分支掩码矩阵用错（已验证）

`main.py` 有两条测试路径，口径不一致：

- 训练结束的路径（`main.py:152-157`）会先 `trainer.args.train_matrix = test_rating_matrix` 再测 ✅
- `--do_eval` 路径（`main.py:125-128`）**直接测**，此时 `args.train_matrix` 仍是 `valid_rating_matrix`（`main.py:85` 设的）❌

后果：valid 目标 item 没有被屏蔽，模型把它排在很前面，挤掉 test 目标。实测差异：

| 路径 | HIT@10 | NDCG@10 | MRR |
|---|---|---|---|
| 训练结束（test_rating_matrix） | 0.06596 | 0.03803 | 0.03252 |
| `--do_eval`（valid_rating_matrix） | 0.06287 | 0.02881 | 0.02140 |

HIT@10 只掉 5%，但 **NDCG@10 掉 24%、MRR 掉 34%** —— 因为被错误保留的候选恰好常年站在第 1 名附近。已用独立脚本（设 `train_matrix=test_rating_matrix` 后调 `trainer.test`）验证：换回正确矩阵后结果与训练结束时完全一致，确认就是这一个变量造成的。

> 修复方式（未擅自修改仓库）：在 `main.py:127` 前加一行 `args.train_matrix = test_rating_matrix`。
> 该缺陷来自上游 S3-Rec 的代码骨架，不是本次改动引入的。

### 8.2 早停 patience 与论文不一致

论文 §C 写"验证 MRR 连续 **50** epoch 不升即停"，但代码 `main.py:140` 对 `DistSAModel` 硬编码 `patience=100`（其他模型才 50）。本次因此多跑了 82 个 epoch（100 → 182）。指标没受影响（权重取的是第 82 epoch 的最优值），但**训练时间翻了一倍**。

### 8.3 指标名不一致但口径相同

论文写 Recall@N，代码里叫 HIT@N；`utils.recall_at_k` 的实现就是标准 Recall（命中即计 1），可直接对比。NDCG 与 MRR 两处实现也与论文描述一致（MRR 在全 12,103 个候选上算，符合论文"rank all items"）。

---

## 9. 为什么只到 ~90%？偏差分析

按可能性排序：

1. **λ（pvn_weight）没有调**。论文 Fig.2(b) 明确显示 Beauty 上 MRR 对 λ 极敏感（横轴取 `0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1 … 1.0`），且曲线"先升后降"，峰区不在端点。本次只用 README 的 `0.005`。**这是最可能的单一原因**——论文自己也说"a properly selected λ can dramatically improve the performance"。
2. **没有任何网格搜索**。论文报告的是在 {32,64}×{50,100}×{1,2,3}层×{1,2,4}头×{0.3,0.5,0.7}dropout×{1e-3,1e-4}lr 上搜完的最优组合；本次只是 README 给的一组。
3. **weight_decay=0.0 不在论文搜索集内**（论文搜 1e-1/1e-2/1e-3）。本次训练后期明显过拟合（见 §7.2），L2 正则缺失可能是次因。
4. **单次运行、单 seed**。论文与本次都没报多 seed 方差，指标本身有约 ±0.3–0.5% 的抖动（从 §7.2 相邻 epoch 的 MRR 波动可见），不足以解释 10%，但会加剧不确定性。
5. **硬件/精度差异**（3090 vs 4060 Laptop、不同 CUDA/cuDNN）——通常影响极小，可排除为主因。

反过来说，**代码实现与论文公式逐条对应（§2），数据统计逐项吻合（§4），最优权重可逐位复现（§7.3）**，所以这不是"跑错了"，而是"没按论文的搜索预算跑"。

---

## 10. 结论与后续建议

**结论**：STOSA 在 Beauty 上可复现——模型可训练、可收敛、可逐位复测，结果显著优于 SASRec（MRR +11.8%），但单配置只达到论文 DT4SR 的水平，与论文 STOSA 值差约 10%。

**下一步建议（按性价比排序）**：

1. **扫 λ ∈ {0, 0.005, 0.01, 0.02, 0.03, 0.04}**，其余不变。6 次 × 约 40 分钟（patience 改 50 后更短）→ 最可能一步补上大半差距。
2. 把 `patience` 改成 50（对齐论文），节省一半时间；顺带补 `weight_decay ∈ {1e-3, 1e-2}` 两组。
3. 补 3 个 seed（如 42/2020/2021）报 mean±std，让结论可统计。
4. 若要复现论文的 RQ3/RQ4/RQ5（冷启 item、序列长度分组、注意力热图、t-SNE），需要在评估时额外按 item 流行度分桶——当前 `trainers.py` 没有这个统计，需要自己写脚本。

---

## 附录 A · 产物清单

| 文件 | 说明 |
|---|---|
| `outputs/DistSAModel-Beauty-64-1-4-gelu-0.0-0.3-100-0.001-0.0-10-1.0-0.005.pt` | 最优权重（epoch 82，6.75 MB，1,684,315 参数） |
| `outputs/DistSAModel-Beauty-64-1-4-gelu-0.0-0.3-100-0.001-0.0-10-1.0-0.005.txt` | 完整日志（84.6 KB，183 epoch 逐轮 loss/指标 + 结尾 `result_info`） |
| `data/Beauty.txt` | 22,363 用户 / 198,502 交互 |
| `STOSA.pdf` | 论文原文（12 页） |

## 附录 B · 复测脚本（绕过 §8.1 的 do_eval 缺陷）

```python
# 与 main.py 训练结束时同口径的测试：显式切到 test_rating_matrix
from utils import get_user_seqs
from datasets import SASRecDataset
from seqmodels import DistSAModel
from trainers import DistSAModelTrainer
from torch.utils.data import DataLoader, SequentialSampler

user_seq, max_item, valid_mat, test_mat, num_users = get_user_seqs(args.data_file)
# ... 构造 args（同 §6 配置）...
model = DistSAModel(args)
trainer = DistSAModelTrainer(model, DataLoader(train_ds, batch_size=args.batch_size),
                             DataLoader(valid_ds, batch_size=100),
                             DataLoader(test_ds, batch_size=100), args)
trainer.load(args.checkpoint_path)
trainer.args.train_matrix = test_mat          # ← 关键一行
scores, result_info, _ = trainer.test('best', full_sort=True)
print(result_info)
```

## 附录 C · 论文与仓库的引用关系

论文 §4 明确说明代码基于 **S3-Rec**（`https://github.com/RUCAIBox/CIKM2020-S3Rec`），本仓库 README 也写了同一句；因此 `main.py`/`trainers.py`/`datasets.py` 里出现的 S3-Rec 遗留结构（如 `pretrain` 相关分支、`--do_eval` 分支）属正常继承，论文复现只需走 `DistSAModel` 分支。
