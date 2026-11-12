# 论文笔记：STOSA —— 用「距离」替掉「点积」的随机自注意力

> **Sequential Recommendation via Stochastic Self-Attention**
> Ziwei Fan, Zhiwei Liu, Alice Wang, Zahra Nazari, Lei Zheng, Hao Peng, Philip S. Yu
> WWW '22 · DOI [10.1145/3485447.3512077](https://doi.org/10.1145/3485447.3512077) · [arXiv:2201.06035](https://arxiv.org/abs/2201.06035)
> 代码：[zfan20/STOSA](https://github.com/zfan20/STOSA)（基于 S3-Rec 改写）

---

## 0. 一句话总结

把 item 从"一个点"换成"一个高斯分布"，把注意力权重从"点积"换成"负 Wasserstein 距离"——
于是**不确定性**（协方差）和**协同传递性**（三角不等式）同时进了模型，再给 BPR 补一项正负样本距离约束。

这份笔记记录我对论文动机、推导和实验的理解，以及几处读的时候觉得值得留意的地方。

---

## 1. 背景：SASRec 这类模型差在哪

序列推荐（SR）把用户历史按时间排成 $S_u=[v_1,\dots,v_{|S_u|}]$，目标是预测下一个 item。
SASRec / BERT4Rec 用 Transformer 的自注意力在序列内部建模 item-item 关系，长期是 SOTA。
STOSA 认为这类做法有三个缺口。

### 1.1 确定性建模丢掉不确定性

现有方法把 item 嵌成**固定向量**，隐含假设"用户兴趣是确定性的"。
但真实序列里 item 转移经常不合逻辑——一本书推荐系统里，兴趣横跨科幻、言情、传记的用户，
本就比只读惊悚、恐怖、奇幻的用户更"不确定"；兴趣波动大的用户更该被建模成一个**分布**而不是一个点。

### 1.2 点积不满足三角不等式，丢了协同传递性

这是论文最漂亮的一处论证。给定三个 item 的嵌入：

| item | 向量 |
|---|---|
| $i_x$ | $(0,2)$ |
| $i_y$ | $(1,1)$ |
| $i_z$ | $(2,0)$ |

观测到了转移对 $(i_x\to i_y)$ 和 $(i_y\to i_z)$，直觉上 $i_x$ 和 $i_z$ 也该接近。但：

- $\langle i_x,i_y\rangle = 2$
- $\langle i_y,i_z\rangle = 2$
- $\langle i_x,i_z\rangle = 0$

没被观测到的 $(i_x,i_z)$ 在点积下直接"断了"。而距离天然满足三角不等式：

$$d(i_x,i_z)\le d(i_x,i_y)+d(i_y,i_z)$$

这正是协同传递性（collaborative transitivity）：**没见过的 item 也能借道中间 item 拿到协同信息**。
对冷启 item 尤其关键——它们本来就缺交互数据，只能靠"传递"活。

论文特别指出前作 DT4SR 的问题：它虽然把 item 表示成分布，但用点积算注意力，
所以**只拿到了不确定性，没拿到传递性**。

### 1.3 BPR 对正负样本之间没有距离约束

BPR 只比"用户对正样本 vs 负样本的打分差"，并不保证正样本在隐空间里真的远离负样本。
当负样本是随机采样（不一定 hard）时，这个缺口会把优化带偏。
所以论文额外加一项 hinge：正样本到负样本的距离，必须比预测距离还大。

---

## 2. 方法

整体三个模块：随机嵌入 → Wasserstein 自注意力 → 带 ℓ_pvn 正则的 BPR。

### 2.1 随机嵌入

每个 item 用**对角高斯**表示 $p_j=\mathcal{N}(\mu_j,\Sigma_j)$，$\Sigma_j$ 是对角阵，
均值和协方差各有一张嵌入表 $M^\mu, M^\Sigma\in\mathbb{R}^{|\mathcal{V}|\times d}$，
并各自配一套位置嵌入：

$$\hat E^\mu_{S_u}=\big[\mu_{s_1}+p^\mu_{s_1},\dots,\mu_{s_n}+p^\mu_{s_n}\big],\qquad
\hat E^\Sigma_{S_u}=\big[\sigma_{s_1}+p^\Sigma_{s_1},\dots,\sigma_{s_n}+p^\Sigma_{s_n}\big]$$

为什么用对角高斯而不是完整协方差？因为完整协方差是 $d\times d$，
参数量和计算量都直接起飞；对角近似把每个维度当作独立的高斯分量，
既保留了"每个维度上的不确定程度"，又能把 $W_2$ 的 trace 项化简成逐元素运算（见下）。

### 2.2 从 $W_2$ 到可计算的形式

两个高斯的 2-Wasserstein 距离定义是：

$$W_2^2(p_1,p_2)=\|\mu_1-\mu_2\|_2^2+
\mathrm{tr}\Big(\Sigma_1+\Sigma_2-2\big(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2}\big)^{1/2}\Big)$$

乍看吓人，但对角情形一下就塌了：$(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2}$ 落在对角线上就是
$\mathrm{diag}(\sqrt{\sigma_{1i}\sigma_{2i}})$，于是 trace 项逐维度变成

$$\sigma_{1i}+\sigma_{2i}-2\sqrt{\sigma_{1i}\sigma_{2i}}=\big(\sqrt{\sigma_{1i}}-\sqrt{\sigma_{2i}}\big)^2$$

合起来就是非常干净的一行：

$$\boxed{W_2^2(p_1,p_2)=\|\mu_1-\mu_2\|_2^2+\sum_i\big(\sqrt{\sigma_{1i}}-\sqrt{\sigma_{2i}}\big)^2}$$

**这就是全部计算量**。均值部分等价于欧氏距离平方（可以用矩阵乘法批量算），
协方差部分就是在"标准差"上再算一次欧氏距离平方。
换句话说，STOSA 并没有比 SASRec 多算什么昂贵的操作，只是把"点积"换成了"对 $(\mu,\sqrt{\sigma})$ 的欧氏距离"。

### 2.3 Wasserstein 自注意力

注意力权重直接取负距离（越大越"近"）：

$$A_{kt}=-W_2^2(s_k,s_t)$$

归一化后按高斯线性组合的性质聚合——**均值用一阶权重、方差用二阶权重**：

$$\tilde A_{kt}=\frac{A_{kt}}{\sum_{j\le t}A_{jt}},\qquad
\mathbf z^\mu_{s_t}=\sum_{k\le t}\tilde A_{kt}V^\mu_k,\qquad
\mathbf z^\Sigma_{s_t}=\sum_{k\le t}\tilde A_{kt}^{2}V^\Sigma_k$$

方差那行看着奇怪，其实是最朴素的概率论：若 $X_k\sim\mathcal{N}(0,\Sigma_k)$ 独立，
则 $\sum_k w_k X_k\sim\mathcal{N}\big(0,\sum_k w_k^2\Sigma_k\big)$——
**权重进方差要平方**。写代码时这一处最容易抄错（写成 $\tilde A$ 就退化成普通加权了）。

因果掩码照旧：$k\le t$，只看过去。

### 2.4 FFN 与正定技巧

自注意力 + 前馈 + 残差 + LayerNorm，结构上和 SASRec 一样，只有一处细节值得记：

$$\mathbf Z^\Sigma_{s_t}=\mathrm{ELU}\Big(\mathbf z^\Sigma_{s_t}+\mathrm{Dropout}(\mathrm{FFN}^\Sigma(\mathrm{LN}(\mathbf z^\Sigma_{s_t})))\Big)+1$$

为什么要 $\mathrm{ELU}(\cdot)+1$？因为协方差必须是**正定**的。
ELU 的下界是 $-1$，所以 $\mathrm{ELU}(x)+1>0$——这是一个又便宜又稳的保正手段，
比 exp 不会溢出，比 ReLU 多了负半轴的斜率（梯度不断）。
论文还提了一句用 ELU 而非 ReLU 是为了数值稳定。

### 2.5 预测层：注意排序方向

打分就是推断出的分布与候选 item 分布之间的 $W_2$：

$$d_{s_t,j}=W_2\big(p_{s_t},p_j\big)$$

**距离越小越可能**，所以 top-N 要按**升序**取。
这一点和点积模型反着来，和 SASRec 代码对照着读时极其容易翻车（指标突然掉一半，八成是这里）。

### 2.6 损失：BPR + ℓ_pvn

$$\mathcal{L}=\sum_{S_u}\sum_{t=1}^{|S_u|}
-\log\sigma\big(d_{s_t,j^-}-d_{s_t,j^+}\big)
+\lambda\big[d_{s_t,j^+}-d_{j^+,j^-}\big]_+
+\beta\|\Theta\|_2^2$$

- 第一项：BPR。注意因为 $d$ 是距离，正样本的距离要更小，所以括号里写成 $d_{j^-}-d_{j^+}$。
- 第二项：$\ell_{pvn}$。语义是"正样本到负样本的距离 $d_{j^+,j^-}$，至少要大于预测距离 $d_{s_t,j^+}$"；
  否则说明正样本反而更靠近负样本，反直觉。hinge 使它在满足条件时归零。
- $\lambda$ 是这一项的唯一旋钮，也是论文里最敏感的旋钮（见 §5.3）。

---

## 3. 为什么是 Wasserstein，不是 KL，也不是点积

| 性质 | 点积 | KL 散度 | Wasserstein |
|---|---|---|---|
| 能度量分布之间的差异 | ✗（只能度量向量） | ✓ | ✓ |
| 满足三角不等式 | ✗ | ✗（而且不对称） | ✓ |
| 分布不重叠时的行为 | — | 容易发散/爆炸 | 依然平滑有界 |

三角不等式保证协同传递性（这是论文的核心论证）；平滑性保证训练稳定。
论文脚注里提到试过 KL，效果更差、推理也更慢，所以最终选了 $W_2$。

---

## 4. 复杂度

设 $n$ 为序列长度、$d$ 为隐维：

- 空间：$O(|\mathcal{V}|d+nd+d^2/2)$，比 SASRec 的 $O(|\mathcal{V}|d+nd+d^2)$ 还略小
- 时间：$O(nd+n^2d+4n^2+nd^2/2)$，与 SASRec 的 $O(n^2d+nd^2)$ 同阶

关键在"公平比较"的设定：**把总维度对半分**给均值与协方差，
即 SASRec 用 $d$，STOSA 就用 $d_\mu=d/2$ 和 $d_\Sigma=d/2$ 各一份。
这样参数量基本打平，性能差异才归因于建模方式而不是"参数更多"。
（读代码时要注意：仓库里的 `--hidden_size` 是**分别**给均值和协方差的宽度，
所以 `hidden_size=64` 实际总维度是 128。）

---

## 5. 实验

### 5.1 设置

- **数据**：Amazon 五个域（Home / Beauty / Tools / Toys / Office），5-core 过滤，leave-one-out 划分：最后一条测试、倒数第二条验证。
- **评估**：对全部 item 排序（不做负采样），Recall@{1,5} / NDCG@5 / MRR。
- **Baseline**：BPR-MF、LightGCN、CML、SML（静态/度量学习）+ TransRec、Caser、SASRec、DT4SR、BERT4Rec（序列）。
- **超参**：embedding {32,64}、seq_len {50,100}、层数 {1,2,3}、头数 {1,2,4}、dropout {0.3,0.5,0.7}、lr {1e-3,1e-4}、L2 {1e-1,1e-2,1e-3}；早停看验证 MRR，连续 50 轮不升则停。

### 5.2 主结果（以 Beauty 为例）

| 模型 | Recall@1 | Recall@5 | NDCG@5 | MRR |
|---|---|---|---|---|
| BPR-MF | 0.0082 | 0.0300 | 0.0189 | 0.0216 |
| LightGCN | 0.0064 | 0.0287 | 0.0174 | 0.0203 |
| TransRec | 0.0085 | 0.0321 | 0.0204 | 0.0236 |
| Caser | 0.0112 | 0.0309 | 0.0214 | 0.0231 |
| SASRec | 0.0129 | 0.0416 | 0.0274 | 0.0291 |
| DT4SR | 0.0143 | 0.0449 | 0.0296 | 0.0323 |
| BERT4Rec | 0.0119 | 0.0396 | 0.0257 | 0.0294 |
| **STOSA** | **0.0193** | **0.0504** | **0.0351** | **0.0360** |

五个数据集上 STOSA 全指标第一，相对次优提升 3.16%–35.11%，MRR 提升 5.68%–11.54%。
提升在 top-1 上最大（Beauty 的 Recall@1 直接 +35.11%），说明"分布表示"对**排序头部**帮助最明显。

几个附带观察：
- 静态方法普遍打不过序列方法 → 时间顺序信息确实重要。
- 度量学习类（CML / SML）反而偏弱，论文归因于它们把嵌入归一化到单位球，限制了表达。
- DT4SR 在 5 个数据集里 3 个超过 SASRec → 分布表示本身有价值，只是缺了传递性。

### 5.3 消融：λ 的敏感性

论文对 $\lambda\in\{0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1,\dots,1.0\}$ 做了扫描，结论：

- $\lambda=0$ 时（即去掉 $\ell_{pvn}$）STOSA 在多数数据集上仍优于 SASRec（Tools 除外）→ **Wasserstein 注意力本身是主要贡献**。
- 随着 $\lambda$ 增大，MRR 先升后降；**曲线很陡**，说明这一项必须调。
- 论文的解释：负样本是采样出来的、不是真正严格负的，$\lambda$ 太大反而被噪声带偏。

### 5.4 冷启与可视化

- **按 item 流行度分桶**：提升最大的组恰好是交互 ≤ 3 的冷启 item，验证协同传递性确实在帮忙。有趣的是 Beauty / Toys 上热门 item 反而略降，论文归因于"热门 item 的随机表示带来了更多噪声邻居"。
- **按用户序列长度分桶**：序列最长的用户组提升最大（9.70%–54.45%），符合"交互多的用户兴趣更分散、更需要不确定性建模"。
- **注意力热图**：SASRec 只点亮少数几个位置，STOSA 的权重分布更均匀 → 更像在"综合"而非"挑一个最能解释的"。
- **t-SNE**：SASRec 学到的热门 item 散落在边缘，STOSA 把它们挤到中心形成一个紧致簇，冷启 item 则均匀散布——论文把这当作传递性生效的直观证据。

---

## 6. 读的时候忍不住记下来的几处

**① "公平比较"的口径要小心。** 附录 A 明确说总维度对半分（$d_\mu=d_\Sigma=d/2$），
但表 1 里又写"STOSA 只搜 {32,64}"。这两句话合起来到底指总维度还是单侧维度，论文没写死；
落到代码上，`hidden_size=64` 是均值和协方差**各** 64。讨论"同参数量"时这一点必须说清楚，否则容易被质疑。

**② 论文自己的统计表有一处小错。** Table 2 中 Beauty 的 avg. interactions 写 8.3，
但 198,502 / 22,363 = 8.88；density 写 0.05%，实际 198,502 / (22,363×12,101) ≈ 0.073%。
其他四个数据集这两列都自洽，只有 Beauty 对不上，像是从 Home 那一行抄过来的。
另外 §5.1 说"more than 1.2 million users"，但 Table 2 五个数据集用户数相加只有约 13 万、交互约 110 万——
大概率是把 interactions 写成了 users。**不影响结论，但引用这些统计量时要留意。**

**③ 早停策略论文与代码不一致。** 论文写"验证 MRR 连续 50 轮不升则停"，
开源代码里 `main.py` 对 DistSAModel 用的是 `patience=100`（其他模型才 50）。
按论文复现时训练时间会凭空翻倍。

**④ 归一化方式论文写得比代码"温和"。** 论文 Eq.(6) 是直接按和归一化 $\tilde A=A/\sum A$，
但代码里在负距离之后还过了一次 softmax（并按 $\sqrt{\text{head\_dim}}$ 缩放）。
两者在负距离上不等价——softmax 会把近邻的优势指数级放大。
论文正文没提这个 softmax，实际影响可能不大，但严格对照时算一处细节差异。

**⑤ 缺显著性。** 全文没有多随机种子的方差、也没有显著性检验；考虑到指标本身的抖动量级
（相邻 epoch 之间 MRR 就会波动 1e-3 上下），"提升 3%"这种量级的结论最好自己复现一次再引用。

---

## 7. 我自己的复现补充（Beauty）

我在本机（Python 3.14 + PyTorch 2.11 + RTX 4060 Laptop）跑了一遍，用的就是 README 给的那组配置
（hidden 64 / 1 层 / 4 头 / dropout 0.3 / lr 1e-3 / λ=0.005 / seq_len 100），结果：

| 指标 | 论文 | 复现 |
|---|---|---|
| Recall@1 | 0.0193 | 0.01637 |
| Recall@5 | 0.0504 | 0.04539 |
| NDCG@5 | 0.0351 | 0.03142 |
| MRR | 0.0360 | 0.03252 |

大约到论文的 **90%**：明显高于 SASRec（MRR +11.8%），但只与论文里的 DT4SR 持平。
最可能的原因是**没做超参搜索**，尤其是 $\lambda$——论文的 Fig.2 已经明示这一项极敏感，
而我只用了 README 的默认值 0.005。另外 `weight_decay=0` 不在论文的搜索范围内，
训练后期明显过拟合（loss 一路降到 0.02，验证指标却从第 82 轮就开始回落）。

想追平论文的话，性价比最高的一步是**扫 $\lambda\in\{0,0.005,0.01,0.02,0.03,0.04\}$**，
其余保持不变。

---

## 8. 一句话回顾

STOSA 的贡献不是"又加了一个模块"，而是**把度量从内积换成了距离**：
一旦换掉，不确定性可以自然地塞进方差里，三角不等式又免费带来了协同传递性，
再把正负样本距离补进损失，三件事就都齐了。
代价也清楚——多记一套协方差嵌入、多一次 $\sqrt{\sigma}$ 的距离计算、多一个必须调的 $\lambda$。

对我来说最值得带走的两个点：

1. **"表示"和"度量"要配套**。既有分布表示又用点积（DT4SR）只能拿到一半收益。
2. **换度量就能换性质**。想要传递性，就别用不满足三角不等式的度量——这个思路可以迁到很多场景。

---

### 参考

- Fan et al., *Sequential Recommendation via Stochastic Self-Attention*, WWW '22.
- Fan et al., *Modeling Sequences as Distributions with Uncertainty for Sequential Recommendation* (DT4SR), CIKM '21.
- Kang & McAuley, *Self-Attentive Sequential Recommendation* (SASRec), ICDM '18.
- Sun et al., *BERT4Rec*, CIKM '19.
- Sun et al., *S3-Rec*（本仓库的代码骨架来源），CIKM '20.
- Clement & Desch, *An elementary proof of the triangle inequality for the Wasserstein metric*, 2008.
