# 本地运行记录 (RUN_LOCAL)

本文件记录：为了让 **PRISM: Personalized Recommendation via Information Synergy Module** 在这台机器上跑起来，
做了哪些改动、数据从哪里来、以及复现命令。

环境：Windows 11 / Python 3.14.0 / torch 2.11.0+cu126 (CUDA 12.6) / RTX 4060 Laptop (8 GB) / numpy 2.4.1。

---

## 1. 仓库里缺什么

仓库只带了代码，没有带 `data/`（`data/` 下只有 4 个预处理脚本）。而 `main.py` 的三个必需输入都指向
`data/Beauty/`：

| 文件 | 作用 | 仓库是否提供 |
|---|---|---|
| `reviews_Beauty.txt` | 交互序列 `user item item ...` | ❌ |
| `image_features_Beauty.pt` | 每个物品 512 维图像特征 | ❌ |
| `text_features_Beauty.pt` | 每个物品 512 维文本特征 | ❌ |

所以「直接 `python main.py`」在原仓库状态下必然失败。

## 2. 代码改动（共 1 处）

`main.py` 顶部原本有 `from InDiRec import InDiRec`，而 `InDiRec.py` 依赖 `faiss`（GPU 聚类），
本机没有 `faiss`，于是**连默认的 STOSA/PRISM 路径都跑不起来**，一 import 就 `ModuleNotFoundError`。

改为把该 import 延迟到真正选择 InDiRec 分支时（`else:` 分支内）。STOSA / SASRec 路径不受影响。

另外为「与论文对比」加了三个**消融/实验开关**（都不影响默认行为）：

| 开关 | 作用 |
|---|---|
| `--no_prism` | 关掉 PRISM 融合分支（Interaction Expert + Adaptive Fusion）及其损失，但 item embedding **仍会加上**投影后的图文向量（本仓库 STOSA 的默认输入形式） |
| `--no_modality` | 连图文向量也不加，退回**纯 ID 的 STOSA**，即论文 Table 2 / Table 5 里 "STOSA (w/o PRISM)" 的含义 |
| `--fusion_only` | 让 PRISM 的融合向量成为模态的**唯一**通路（不再额外加原始图文向量），用于检验「重复计入模态」是否稀释了 PRISM 的贡献 |

实现位于 `STOSA.py` 的 `use_prism` / `use_modality` / `fusion_only` 三个标志与
`add_position_{mean,cov}_embedding` 的分支，`main.py` 增加对应的三个 argparse 参数。

同时修掉了两个会让 λ 失效的 bug（详见 9.1）：`PRISM.py` 原来把四个交互损失权重写死成 0.1，
`main.py` 又把 `--lambda_*` 声明成 `type=int`（传浮点会直接报错）。两处都改了。

> 注：`SASRec.py` 的 `replace_embedding()` 引用了 `self.image_mean_embeddings` 等 4 个
> **从未在 `__init__` 中定义的属性**，因此 `--model_name SASRec` 目前是不可用的（会 `AttributeError`）。
> 论文默认 backbone 是 STOSA，本次运行未走该分支，故未改动。

## 3. 数据准备

原始数据（Amazon 2014 5-core Beauty）取自 HuggingFace 镜像 `ttb06/amazon_beauty_5_2014`
（`reviews_Beauty_5.json.gz`, 42.7 MB）。斯坦福原站 `snap.stanford.edu` 实测只有 ~1.5 KB/s，不可用。

`data_local/prepare_data.py` 复刻了 `data/DataProcessing.py` 的 5-core 过滤逻辑：

```
22,363 users / 12,101 items / 198,502 interactions / avg length 8.88
```

与 STOSA、SASRec 等论文里 Amazon Beauty 的标准统计量完全一致，说明数据切分正确。

> 位置说明：数据放在新建的 `data_local/` 而不是仓库自带的 `data/`。
> 原因：本次会话的写入沙箱对 `data/`、`images/` 这两个**已存在**的子目录没有正确授权
> （Windows 低完整性标签 + 沙箱写权限只落在工作区根目录，未传播到已存在的子目录），
> 新建目录则正常继承。为不改动盘上的访问控制，改用新建目录 + 命令行参数覆盖路径。

## 4. 多模态特征的当前状态（重要）

`image_features_Beauty.pt` / `text_features_Beauty.pt` 现在是**真实的 CLIP ViT-B/32 特征**
（512 维，L2 归一化），由 `data_local/gen_clip_features.py` 生成：

| 模态 | 覆盖率 | 来源 |
|---|---|---|
| 文本 | **99.9%** (12,094 / 12,101) | 5-core 元数据 `text`（title + description + brand + categories 合并） |
| 图像 | **99.8%** (12,079 / 12,101) | 元数据 `imUrl` → Amazon 媒体 CDN 实际下载 12,079 张图 |

未覆盖的极少数物品沿用 `process_clip.py` 自己的兜底：`N(0, 0.02)` 噪声向量。

### 这一步原本卡在哪、怎么绕过去的

README 要求 `pip install clip` + 下载商品图，三个环节都受阻，逐条绕开：

1. **`pip` 在本沙箱内不可用**：任何 `pip install` 都在解包 wheel 时写 `.whl.metadata` 报
   `PermissionError [WinError 5]`（把 `TEMP` 换到工作区、加 `--no-cache-dir` 均无效，而普通 Python
   原地写同名文件却正常）。→ 绕过 pip：**wheel 就是 zip**，用 `urllib` 下载后 `zipfile.extractall()` 解包。
2. **`git` 可用**（走自带的 OpenSSL，不像 PowerShell/curl 那样卡在 `schannel: SEC_E_NO_CREDENTIALS`）
   → `git clone --depth 1 https://github.com/openai/CLIP`，把仓库目录塞进 `sys.path` 直接用，
   不做 `pip install -e`；它只缺 `ftfy`，单独解包一个 wheel 即可。
3. **CLIP 权重**：`ViT-B-32.pt`（337.6 MB）从 `openaipublic.azureedge.net` 下载，约 4 分钟。
4. **2014 年的商品图链接其实还活着**：`imUrl` 形如
   `http://ecx.images-amazon.com/images/I/<id>.jpg`，重写成
   `https://m.media-amazon.com/images/I/<id>.jpg` 后 24 线程下载，
   **12,079 / 12,080 成功，仅 1 张失败**。

顺带说明：之前尝试直接复用公开的 CLIP 特征（HF `SLKpnu/amazon_2014_5_core` 的
`idx_*_emb_map_clip.pt`，正好也是 512 维）失败了——它和本 Beauty 5-core 的 asin
只重合 **11 / 12,101**，无法对齐。所以这次是自己算的。

### 复现

```powershell
python .\data_local\fetch.py                 # reviews_Beauty_5.json.gz
python .\data_local\prepare_data.py --dataset Beauty --feature-source noise   # 交互 + itemmap.json
python .\data_local\gen_clip_features.py --workers 24 --batch 128             # 真实 CLIP 特征（约 10 分钟）
```

依赖目录：`_vendor/clip_repo`（openai/CLIP 源码）、`_vendor/site`（ftfy）、
`_vendor/ViT-B-32.pt`（权重）、`data_local/Beauty/images/`（商品图缓存，约 0.6 GB）。

## 5. 复现命令

```powershell
# 1) 生成交互文件 + 占位特征
python .\data_local\prepare_data.py --dataset Beauty --feature-source noise

# 2) 训练 + 验证 + 测试（真实 CLIP 特征）
python main.py `
  --data_dir ./data_local/Beauty/ `
  --data_name reviews_Beauty `
  --image_emb_path ./data_local/Beauty/image_features_Beauty.pt `
  --text_emb_path ./data_local/Beauty/text_features_Beauty.pt `
  --output_dir ./outputs/Beauty_clip `
  --epochs 50 --patience 10
```

日志/指标写进 `outputs/<run>/STOSA-reviews_Beauty-256-0.1-0.1-0.1.txt`，checkpoint 写进同目录 `.pt`。
`outputs/Beauty_clip` 是真实 CLIP 特征那一版，`outputs/Beauty_placeholder` 是占位特征那版（第 6 节的对照）。

## 6. 结果（实测，A/B 对照）

两组唯一的区别就是 `*_features_Beauty.pt` 的内容，其余超参完全一致
（`--epochs 50 --patience 10`，35,793,759 参数，峰值显存 2.17 GB）。
单 epoch：训练 88 个 batch 约 17 s，验证 224 个 batch 约 12.5 s。

**测试集（best checkpoint）**

| 特征 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | 耗时 |
|---|---|---|---|---|---|
| 占位噪声 | 0.0730 | 0.0994 | 0.0452 | 0.0518 | 26:40 |
| **真实 CLIP** | **0.0750** | **0.1032** | **0.0463** | **0.0534** | 25:56（epoch 48 早停） |
| 相对提升 | +2.7% | +3.8% | +2.4% | +3.1% | 收敛更快 |

**验证集**

| 特征 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | best epoch |
|---|---|---|---|---|---|
| 占位噪声 | 0.0929 | 0.1226 | 0.0582 | 0.0656 | 43 |
| **真实 CLIP** | **0.0948** | **0.1274** | **0.0593** | **0.0675** | 38 |

收敛速度差异更明显：同样的 Recall@10 水平，真实特征大约少用 6～8 个 epoch
（epoch 6：0.0689 vs 0.0590；epoch 15：0.0884 vs 0.0819）。

**怎么解读**：真实图文信号确实带来了稳定但幅度不大的增益，而且收敛更快——这与
「Amazon Beauty 上 ID embedding 占主导、模态信息是补充」的普遍观察一致。
提升幅度只有 2%～4%，**不能拿来和论文表格对比**：评测沿用 `trainers.py` 自身实现
（全排序后只取 40 个候选再算 Recall/NDCG），与论文的全量排序协议不完全一致；
也没有做多 seed 平均，2% 量级在这个方差下属于「方向对、幅度待确认」。

顺带一提：早停触发时 `rec_cur_loss` 已经变成负值（如 `-0.0788`），这是
`PRISM.py` 里 `synergy_loss` 用余弦相似度求和、可为负导致的，不是 bug。



## 7. 其它可复现的坑

- `--model_name InDiRec`：需要 `faiss`（`pip install faiss-cpu`，本机 cp314 无预编译轮子，会尝试源码编译），
  另外 `InDiRecTrainer` 里用了 `self.seq_model`、`self.optimizer`、`self.args.p`、`self.args.noise_ratio`、
  `self.args.num_cluster`、`self.args.print_log_freq`，而 `main.py` 的 argparse 没有定义这些，该分支目前不完整。
- `data/DataProcessing.py` 里 `DATASET` 是硬编码的，且 `main.py` 期望文件名是 `reviews_<name>.txt`，
  与脚本产出的 `<name>.txt` 不一致，需要自行改名。
- `data/process_clip.py` 的 itemmap 基于 **meta 文件全量**物品编号，而模型 embedding 大小是
  `max_item(reviews)+2`；直接套用会出现长度不匹配，需要按 reviews 的 id 重新对齐。

## 8. 与论文 Table 2 / Table 5 的对比（Beauty + STOSA）

### 8.1 论文原数据（PRISM.pdf, Table 2，Beauty 列）

| 模型 | R@10 | R@20 | N@10 | N@20 |
|---|---|---|---|---|
| SASRec | 0.0534 | 0.0839 | 0.0247 | 0.0332 |
| **STOSA（vanilla）** | **0.0648** | **0.0941** | **0.0339** | **0.0385** |
| PRISM + SASRec | 0.0681 | 0.1051 | 0.0343 | 0.0457 |
| **PRISM + STOSA** | **0.0762** | **0.1083** | **0.0481** | **0.0549** |
| InDiRec | 0.0941 | 0.1286 | 0.0573 | 0.0660 |
| PRISM + InDiRec | 0.1012 | 0.1388 | 0.0618 | 0.0711 |

论文自报的 vs. Vanilla（PRISM+STOSA 相对 STOSA）：**+17.59% / +15.09% / +41.89% / +42.60%**

### 8.2 本地四组实验（同一套代码，只切开关，测试集）

| 变体 | R@10 | R@20 | N@10 | N@20 | 训练时长 |
|---|---|---|---|---|---|
| ① 纯 ID STOSA（`--no_modality`） | 0.0724 | 0.0985 | 0.0421 | 0.0486 | 13:54（29 ep 早停） |
| ② + 图文直加，无融合（`--no_prism`） | 0.0735 | 0.1011 | 0.0454 | 0.0524 | 24:56（50 ep） |
| ③ + PRISM，占位噪声特征 | 0.0730 | 0.0994 | 0.0452 | 0.0518 | 26:40（50 ep） |
| ④ **+ PRISM，真实 CLIP 特征** | **0.0750** | **0.1032** | **0.0463** | **0.0534** | 25:56（49 ep 早停） |

相对增益（相对 ① 纯 ID，等价于论文的 "vs. Vanilla"）：

| 变体 | R@10 | R@20 | N@10 | N@20 |
|---|---|---|---|---|
| ② 只加模态 | +1.52% | +2.64% | +7.84% | +7.82% |
| ④ PRISM + 真实 CLIP | **+3.59%** | **+4.77%** | **+9.98%** | **+9.88%** |

其中「PRISM 融合本身」的净贡献（④ 相对 ②）是 **+2.04% / +2.08% / +1.98% / +1.91%**。

### 8.3 结论：绝对水平复现了，相对提升没复现

**（1）绝对水平：吻合得很好**

| 指标 | 本地 ④ | 论文 PRISM+STOSA | 差距 |
|---|---|---|---|
| R@10 | 0.0750 | 0.0762 | −1.6% |
| R@20 | 0.1032 | 0.1083 | −4.7% |
| N@10 | 0.0463 | 0.0481 | −3.7% |
| N@20 | 0.0534 | 0.0549 | −2.7% |

单 seed、单卡 4060 Laptop、50 epoch 以内，能落在论文数值的 2%~5% 内，说明**数据切分、特征、
评测口径和整体实现是对得上的**。

**（2）相对提升：明显小于论文**

- 本地：PRISM 相对纯 ID 基线 +3.59%（R@10）、+9.98%（N@10）
- 论文：+17.59%（R@10）、+41.89%（N@10）
- 差 5~12 倍。

**（3）最大原因是基线强度不同**

我的纯 ID STOSA 基线是 0.0724，论文的 STOSA 是 0.0648 —— **我的基线本身就比论文高 11.7%**。
基线越强，留给 PRISM 的提升空间越小，所以绝对水平接近、相对 gap 被压缩。
这也解释了为什么「只把 CLIP 向量直接加进 item embedding」（②）就已经吃掉了大部分多模态收益，
PRISM 的专家融合在其之上只剩约 +2%。

### 8.4 造成差异的可核对原因

| 差异点 | 论文 | 本地 |
|---|---|---|
| 随机种子 | 5 次平均 + paired t-test (p<0.05) | 单 seed（42） |
| 交互损失权重 λ | 网格搜索 {0.01, 0.05, 0.1, 0.2, 0.5, 1.0} | 原仓库写死 0.1（已修，见 9.1）；本地用 0.1 与 (0.05, 0.2, 0.5) 两组 |
| 训练轮数 | 未在正文给出（RecBole 常规均为数百 epoch） | 29~49 epoch 早停（另有 150 epoch 上限的长训，见 9.3） |
| 框架 | RecBole 2.0 | 本仓库自研实现 |
| 结构超参 | 未给出 | hidden 256 / 1 层 / 4 头 / seq 100 / batch 256（另有 64/50 的「论文量级」配置，见 9.0） |
| 骨干 | 三个（SASRec / STOSA / InDiRec，主力是 InDiRec 0.1012） | 只跑 STOSA |
| 硬件 | RTX 4090 | RTX 4060 Laptop |

> **曾经的一个实打实的仓库 bug（已修）**：`PRISM.py` 原来把四个权重写死成
> `self.lambda_uni_v = self.lambda_uni_t = self.lambda_syn = self.lambda_red = 0.1`，
> **从不读取 `args.lambda_*`**；而 `main.py` 又把这几个参数声明成 `type=int`（传 0.5 会直接报错）。
> 两处都已修正，现在 `--lambda_uni/--lambda_syn/--lambda_red` 真正生效。

### 8.5 效率对比（Table 5, Beauty, STOSA）

| 项 | 论文 STOSA w/o → w/ PRISM | 本地 ① → ④ |
|---|---|---|
| 每 epoch 训练时间 | 7.40 → 11.88 s（+60.5%） | ≈15.6 → ≈18.3 s（+17.3%，估算） |
| 显存 | 2.27 → 2.60 GB（+14.5%） | 1.94 → 2.17 GB（+11.9%） |
| 参数量 | — | 34.74M → 35.79M（PRISM +1.05M） |

两点口径说明：
1. **绝对耗时不可比**（4090 桌面卡 vs 4060 Laptop，且论文未给 batch/seq_len），只有「PRISM 带来的相对开销」可比；
2. **显存口径不同**：本仓库 `test_real_memory_usage()` 测的是**单个 batch 前反向的峰值**，
   不是论文那种整 epoch 峰值，所以只能内部横比（+11.9% 与论文 +14.5% 量级一致）。
   每 epoch 训练时间是「总时长 − 验证 − 测试」估算出来的。

## 9. 尝试缩小「相对提升」差距（5–12 倍）

第 8 节发现：绝对水平对得上，但相对提升只有论文的 1/3～1/4。本节记录四组针对性实验。

### 9.0 先修正对比口径：配置对齐

第 8 节用的本仓库默认配置（hidden 256 / seq 100），其**纯 ID 基线就有 0.0724**，
比论文 STOSA 的 0.0648 高 11.7%——基线越强，相对提升空间越小。

于是把配置降到论文量级（hidden 64 / seq 50，其余不变）重跑：

| 配置 | 纯 ID STOSA（测试 R@10） |
|---|---|
| 论文 | 0.0648 |
| 本仓库 64 / 50 | **0.0639**（−1.4%，R@20 −1.6%） |
| 本仓库 256 / 100 | 0.0724（+11.7%） |

**基线成功复现。** 后续所有对比都用 64/50 这组，才是同口径。

### 9.1 修正：λ 硬编码 bug

`PRISM.py` 原本写死 λ=(0.1,0.1,0.1,0.1)，且 `main.py` 把 λ 声明为 `type=int`。
两处都已修（改成读 `args.lambda_*`、`type=float`）。修复后立即生效：
同样 1 个 epoch，λ=0.1 的总损失 0.9067，λ=1.0 变成 0.8291。

> 顺带一个发现：λ=1.0 时总损失会**被推成负数**（`synergy_loss` 用余弦相似度求和，最小化它可以
> 靠"让两个表示互相远离"来白拿，并不改善推荐）。这印证了论文所说「λ 过大会破坏训练」，
> 也说明这个损失函数有被钻空子的空间。

### 9.2 三个杠杆：两个被否掉

同样 64/50 配置，测试集 R@10：

| 实验 | R@10 | 相对纯 ID（0.0639） | 结论 |
|---|---|---|---|
| 纯 ID STOSA（基线） | 0.0639 | — | — |
| PRISM，默认相加接线 + λ=0.1 | **0.0673** | **+5.3%** | 当前最好 |
| PRISM，λ=(0.05, 0.2, 0.5)（论文最优区间） | 0.0651 | +1.9% | ❌ 没帮上忙 |
| PRISM，`--fusion_only`（融合向量独占模态通路） | 0.0529 | −17.2% | ❌ 明显崩了 |
| PRISM，fusion_only + 调 λ | 0.0528 | −17.4% | ❌ 更差 |

**为什么 `--fusion_only` 会崩**：本仓库里 `Adaptive_Fusion_Layer` 先做
`id_emb = item_embeddings.mean(dim=1)`，即把整条序列压成**一个用户级向量**，
再 broadcast 加到每一个时间步上；而原始图文向量是**逐物品**的。
拿一个用户级向量去替代逐物品模态信息，信息量差一个数量级，所以掉得很厉害。

这也正好解释了 PRISM 在本仓库里为什么只值 +2%～+5%：**它加的只是序列级偏置项**，
逐物品的模态信号早被"图文直加"那条路吃掉了。

### 9.3 第四个杠杆：训练长度

同样 64/50 配置，PRISM 训练到 111 epoch 才早停（上限 150，patience 40）：

| 训练长度 | 测试 R@10 | 测试 R@20 | 测试 N@10 | 测试 N@20 |
|---|---|---|---|---|
| 48 epoch（patience 10） | 0.0673 | 0.0896 | 0.0410 | 0.0466 |
| 111 epoch（patience 40） | 0.0674 | 0.0939 | 0.0420 | 0.0487 |

R@10 完全没动，R@20/N@10/N@20 有 1%～4% 的小幅提升。**训练长度不是主要原因。**
（验证集上从 epoch 45 的 0.0834 慢慢爬到 epoch 101 的 0.0873，但这点提升没传递到测试集。）

### 9.4 四轮尝试的净结果

以复现了论文基线的 64/50 配置为准：

| 指标 | 纯 ID 基线 | 最好的 PRISM | 相对提升 | 论文相对提升 | 差距 |
|---|---|---|---|---|---|
| R@10 | 0.0639 | **0.0674** | **+5.5%** | +17.6% | 3.2× |
| R@20 | 0.0926 | 0.0939 | +1.4% | +15.1% | 10.8× |
| N@10 | 0.0366 | 0.0420 | +14.8% | +41.9% | 2.8× |
| N@20 | 0.0437 | 0.0487 | +11.4% | +42.6% | 3.7× |

开始时是 5～12 倍的差距，四轮之后降到约 3 倍（R@20 除外，它本身在本仓库里就不稳定）。
能确认变好的只有 N@10/N@20，R@10 提升有限。

### 9.5 还没排除的可能原因

1. **融合向量的插入位置**。论文 §2.4 只写「integrate them with the ID embedding `e_id`」，
   没写加在哪一层。本仓库加在**嵌入层**（每个时间步都加一次），
   另一种读法是加到**编码器输出的最终偏好表示**上。后者信号更干净，值得一试。
2. **论文的 STOSA 基线是否被充分调过**。论文对 PRISM 做了 λ 网格搜索，对基线则沿用
   原论文配置。若基线欠调，PRISM 的相对提升会被放大。
3. **5 seed + t-test** vs 我的单 seed，2%～5% 量级的提升需要方差才能定论（见 9.6）。

### 9.6 多 seed 验证：提升是真的，但小

64/50 配置，各跑 3 个 seed（42 / 1 / 2），测试集：

| 指标 | 纯 ID | PRISM | 相对 | t 值 | 判定 |
|---|---|---|---|---|---|
| R@10 | 0.0630 ± 0.0011 | 0.0665 ± 0.0011 | **+5.61%** | 3.92 | 显著（p≈0.017） |
| R@20 | 0.0911 ± 0.0020 | 0.0907 ± 0.0012 | −0.37% | −0.25 | 不显著 |
| N@10 | 0.0357 ± 0.0011 | 0.0406 ± 0.0005 | **+13.62%** | 7.09 | 显著 |
| N@20 | 0.0428 ± 0.0013 | 0.0467 ± 0.0003 | **+9.20%** | 5.19 | 显著 |

单 seed 的 R@10 分别是 ID-only `0.0639 / 0.0618 / 0.0632`，PRISM `0.0673 / 0.0670 / 0.0652`
—— seed 间标准差约 0.0011，而两者的差距是 0.0035，**约为噪声的 3 倍**，所以不是运气。

一个值得注意的细节：**R@20 完全没有提升（−0.4%）**。PRISM 把正确答案在 top-10 内排得更靠前了
（Recall@10、NDCG 都涨），但没扩大 top-20 的覆盖面。论文报告的是 R@20 也 +15%，
这一项在本仓库里复现不出来。

### 9.7 结论

四轮尝试后：

| 指标 | 本地（3 seed 均值） | 论文 | 差距倍数 |
|---|---|---|---|
| R@10 | +5.6% | +17.6% | 3.1× |
| R@20 | −0.4% | +15.1% | 无法比较 |
| N@10 | +13.6% | +41.9% | 3.1× |
| N@20 | +9.2% | +42.6% | 4.6× |

- **差距从最初的 5–12 倍缩小到约 3–4.6 倍**，但没能追平；
- 方向是对的、且统计显著：PRISM 确实有正向贡献，只是在本仓库 / 本数据 / 本配置下只有 **+5.6% R@10 / +13.6% N@10**；
- 被否掉的三个杠杆：λ 调参（无效）、融合向量独占模态通路（大幅变差）、延长训练（无效）；
- 剩下最可能的解释是**融合向量的插入位置**（论文 §2.4 措辞含糊）以及**论文基线可能欠调**，
  另外论文的 R@20 提升在本仓库完全复现不出来，这一点本身值得怀疑。

复现命令见第 5 节，把 `--hidden_size 64 --max_seq_length 50` 加上即为本文的「论文量级」配置；
消融再加 `--no_modality`（纯 ID）或 `--no_prism`（无融合）。
