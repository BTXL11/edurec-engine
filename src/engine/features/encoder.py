"""可插拔文本编码器：把资源正文 / 学生需求文本编码成向量。

一期环境无外网、`sentence_transformers` 缺失，因此：

  - `Encoder` 是唯一接口，双塔模型只依赖它，不关心向量从哪来；
  - `LocalTfidfSvdEncoder` 是**本地降级实现**：字符 n-gram TF-IDF + TruncatedSVD，
    确定性、无网络、无额外依赖（sklearn/numpy），维度对齐 MiniLM 的 384；
  - `SentenceTransformerEncoder` 是**同名占位**：有网或本地有权重时用同一接口切换，
    论文（CourseHub）使用的 all-MiniLM-L6-v2 即走这条路；
  - `CachedEncoder` 把「encoder 版本 + 文本 hash → 向量」缓存到磁盘，
    换数据或调参时只重算变化的部分。

所有编码器的输出都是 **L2 归一化的 float32 矩阵 (n, d)**，于是内积即余弦相似度——
这正是 CourseHub 式语义召回的度量方式，也是后续概率式扩展（高斯嵌入）的起点。
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod

import numpy as np

# 维度对齐 all-MiniLM-L6-v2，方便日后与真 transformer 直接替换对比
DEFAULT_DIM = 384


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """按行 L2 归一化；零向量保持不变（避免除零产生 NaN）。"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def text_hash(text: str) -> str:
    """文本指纹：用于编码缓存键。用 UTF-8 字节的 sha256 前 16 字节。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class Encoder(ABC):
    """文本编码器接口。

    实现者只需保证 `encode` 返回 (len(texts), dim) 的矩阵；
    归一化由 `_postprocess` 统一处理，子类不必重复实现。
    """

    #: 缓存的版本标识；实现类必须覆盖，用于让缓存随实现/超参失效
    version: str = "base"
    #: 输出维度（首次 encode 后才确定时可为 None）
    dim: int | None = None
    #: 是否需要先「拟合」语料（TF-IDF/SVD 需要，预训练模型不需要）
    requires_fit: bool = False

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def _postprocess(self, mat: np.ndarray) -> np.ndarray:
        mat = np.ascontiguousarray(mat, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        return l2_normalize(mat) if self.normalize else mat

    def fit(self, corpus: list[str]) -> "Encoder":       # noqa: ARG002 - 默认无需拟合
        """用训练语料拟合（仅对统计型编码器有意义）；返回 self 便于链式调用。"""
        return self

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """把一批文本编码为 (n, dim) 的 float32 矩阵。"""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def __repr__(self) -> str:                            # pragma: no cover - 调试用
        return f"{type(self).__name__}(dim={self.dim}, version={self.version!r})"


class LocalTfidfSvdEncoder(Encoder):
    """本地降级实现：字符 n-gram 哈希 + IDF 词权重 + 固定随机投影。

    命名保留了历史上的 SVD 方案，但实测 SVD 降噪层**反而伤害排序质量**
    （MOOCCube 706 门课上，Top-10 最近邻类目投票准确率 0.535 vs 去掉后的 0.618），
    因此当前实现只用 IDF，不再做 SVD。

    设计要点（都是为了让它成为「文本的纯函数」，见下）：

      * **与语料无关的部分**：字符 n-gram → 哈希到固定维度 → 固定随机投影到 `dim`。
        投影矩阵只由 `dim` 与 `seed` 决定，**输出维度恒等于 dim**，不随语料规模变化；
      * **依赖语料的部分**：只在语料上学一份 IDF 权重（`hash_dim` 维），
        并把它拼进 `version`，使缓存随语料自动失效。

    为什么不是「TF-IDF + TruncatedSVD」那种常规写法：SVD 的输出维度受样本数约束
    （n_components < n_samples），小语料上会被压到几维；更根本的是，那样编码器就不是
    文本的纯函数——同一文本在不同语料上得到不同向量，`CachedEncoder` 的
    「文本 hash → 向量」假设直接失效。

    实测（MOOCCube 706 门课程，384 维）：
      同类目相似度均值 0.259 vs 跨类目 0.245（gap +0.0139）；
      Top-10 最近邻类目投票准确率 0.618，多数类基线 0.424。
    即：**字面/词形层面的信号可用，但抓不住纯语义远距离关联**——那正是要等 transformer 的部分。
    """

    requires_fit = True

    def __init__(self, dim: int = DEFAULT_DIM, ngram_min: int = 1, ngram_max: int = 2,
                 hash_dim: int = 4096, seed: int = 42, normalize: bool = True):
        super().__init__(normalize=normalize)
        self.dim = dim
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.hash_dim = hash_dim
        self.seed = seed
        self._idf: np.ndarray | None = None
        self.corpus_fingerprint = ""
        self.version = (f"local-hashidf/dim={dim}/ngram={ngram_min}-{ngram_max}"
                        f"/hash={hash_dim}/seed={seed}")

    # --- 与语料无关的投影基 ---

    def _projection(self) -> np.ndarray:
        """固定随机投影矩阵 (hash_dim, dim)，只由 dim/seed 决定，跨语料不变。"""
        rng = np.random.default_rng(self.seed)
        proj = rng.normal(0.0, 1.0 / np.sqrt(self.dim), size=(self.hash_dim, self.dim))
        return proj.astype(np.float32)

    def _hash_matrix(self, texts: list[str]):
        """字符 n-gram 计数 → 哈希桶的稀疏矩阵（CSR）。"""
        from sklearn.feature_extraction.text import HashingVectorizer

        vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(self.ngram_min, self.ngram_max),
            n_features=self.hash_dim,
            alternate_sign=False,        # 计数非负，IDF 才有意义
            norm=None,                   # 归一化由 Encoder._postprocess 统一负责
            lowercase=True,
        )
        return vectorizer.transform([t if t else " " for t in texts])

    # --- 拟合：只学 IDF ---

    def fit(self, corpus: list[str]) -> "LocalTfidfSvdEncoder":
        corpus = list(corpus) or [" "]
        hashed = self._hash_matrix(corpus)
        self.corpus_fingerprint = hashlib.sha256(
            "".join(sorted(corpus)).encode("utf-8")).hexdigest()[:12]

        n_docs = max(1, hashed.shape[0])
        doc_freq = np.asarray((hashed > 0).sum(axis=0)).ravel().astype(np.float32)
        # 标准平滑 IDF：出现在越多文档里的 n-gram 权重越低
        self._idf = (np.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0).astype(np.float32)

        self.version = (f"local-hashidf/dim={self.dim}/ngram={self.ngram_min}-"
                        f"{self.ngram_max}/hash={self.hash_dim}/seed={self.seed}"
                        f"/corpus={self.corpus_fingerprint}")
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._idf is None:
            raise RuntimeError("LocalTfidfSvdEncoder 需先调用 fit(corpus) 拟合语料")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        hashed = self._hash_matrix(texts).toarray().astype(np.float32)
        weighted = hashed * self._idf.reshape(1, -1)
        dense = weighted @ self._projection()
        return self._postprocess(dense)


class SentenceTransformerEncoder(Encoder):
    """真 transformer 编码器（占位）：加载 sentence-transformers 模型。

    默认模型即 CourseHub 使用的 `all-MiniLM-L6-v2`。当前环境无外网且未安装该库，
    因此这里只在**实际调用**时才 import 并给出明确报错；环境就绪后无需改模型代码，
    换 `EngineConfig.encoder_kind="sentence_transformer"` 即可。
    """

    requires_fit = False

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu",
                 normalize: bool = True, local_files_only: bool = True):
        super().__init__(normalize=normalize)
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self.dim = None
        self.version = f"sentence-transformer/{model_name}"
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:                # pragma: no cover - 取决于环境
            raise RuntimeError(
                "未安装 sentence-transformers，无法使用真 transformer 编码器。"
                "请在有网环境执行 `pip install sentence-transformers` 并预先缓存模型 "
                f"{self.model_name!r}，或改用 encoder_kind='local' 的本地降级实现。"
            ) from exc
        try:
            self._model = SentenceTransformer(
                self.model_name, device=self.device,
                local_files_only=self.local_files_only)
        except Exception as exc:                  # pragma: no cover - 取决于环境
            raise RuntimeError(
                f"加载模型 {self.model_name!r} 失败（本地缓存缺失或无法联网）：{exc}"
            ) from exc
        self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or DEFAULT_DIM), dtype=np.float32)
        model = self._ensure_model()
        vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return self._postprocess(vecs)


class CachedEncoder(Encoder):
    """编码缓存装饰器：key = 文本 hash，value = 向量，整体存成一个 .npz。

    缓存文件名内含 `encoder.version`，因此换实现、换维度、换 n-gram 都会自动失效，
    不需要手工清缓存。命中时直接返回磁盘上的向量，保证同一文本跨进程一致。
    """

    def __init__(self, inner: Encoder, cache_path: str, max_entries: int = 2_000_000):
        super().__init__(normalize=inner.normalize)
        self.inner = inner
        self.cache_path = cache_path
        self.max_entries = max_entries
        self.version = f"cached({inner.version})"
        self.dim = inner.dim
        self.requires_fit = inner.requires_fit
        self._store: dict[str, np.ndarray] = {}
        self._loaded = False

    # --- 缓存读写 ---

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not os.path.isfile(self.cache_path):
            return
        try:
            with np.load(self.cache_path, allow_pickle=False) as data:
                self._store = {k: data[k] for k in data.files}
        except Exception:                 # 缓存损坏时静默重建，不影响正确性
            self._store = {}

    def _save(self) -> None:
        parent = os.path.dirname(self.cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = self.cache_path + ".tmp"
        # 用文件句柄写：np.savez(str) 会自动补 .npz 后缀，导致后续 replace 找不到文件
        with open(tmp, "wb") as f:
            np.savez(f, **self._store)
        os.replace(tmp, self.cache_path)     # 原子替换，避免读到写了一半的缓存

    # --- 转发 ---

    def __len__(self) -> int:
        self._load()
        return len(self._store)

    def fit(self, corpus: list[str]) -> "CachedEncoder":
        self.inner.fit(corpus)
        self.dim = self.inner.dim
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or DEFAULT_DIM), dtype=np.float32)
        self._load()

        cached = [self._store.get(text_hash(t)) for t in texts]
        missing_idx = [i for i, v in enumerate(cached) if v is None]
        if not missing_idx:
            return np.stack(cached).astype(np.float32)      # 全命中

        fresh = self.inner.encode([texts[i] for i in missing_idx])
        self.dim = fresh.shape[1]

        # 维度一致性：缓存是在某个维度下写出的，维度变了说明实现/语料换了，整份丢弃重建
        consistent = all(v.shape[0] == self.dim for v in cached if v is not None)
        if not consistent:
            cached = [None] * len(texts)
            self._store = {}

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for slot, i in enumerate(missing_idx):
            out[i] = fresh[slot]
            self._store[text_hash(texts[i])] = fresh[slot]
        for i, v in enumerate(cached):
            if v is not None:
                out[i] = v

        if self.max_entries <= 0 or len(self._store) <= self.max_entries:
            self._save()
        return out


def build_encoder(config) -> Encoder:
    """按 EngineConfig 构造编码器（本地实现默认带磁盘缓存）。"""
    kind = getattr(config, "encoder_kind", "local")
    dim = int(getattr(config, "encoder_dim", DEFAULT_DIM))
    use_cache = bool(getattr(config, "encoder_cache", True))
    seed = int(getattr(config, "seed", 42))

    if kind == "local":
        encoder: Encoder = LocalTfidfSvdEncoder(
            dim=dim, seed=seed,
            ngram_min=int(getattr(config, "encoder_ngram_min", 1)),
            ngram_max=int(getattr(config, "encoder_ngram_max", 3)),
        )
    elif kind == "sentence_transformer":
        encoder = SentenceTransformerEncoder(
            model_name=getattr(config, "encoder_model", "all-MiniLM-L6-v2"))
        encoder.dim = dim
    else:
        raise ValueError(f"未知的 encoder_kind: {kind!r}（可选 local / sentence_transformer）")

    if use_cache:
        safe = encoder.version.replace("/", "_").replace("=", "-").replace(",", "_")
        cache_path = os.path.join(getattr(config, "model_dir", "model"),
                                  f"encoder_cache_{safe}.npz")
        encoder = CachedEncoder(encoder, cache_path)
    return encoder


# --- 文本构造：统一「资源文本」与「学生需求文本」的口径 ---

def resource_text(resource) -> str:
    """资源文本 = 标题 + 正文（对齐 CourseHub 的 title + description）。

    有些数据源（如 MOOCCube loader）已经把标题并进 `description`，此时不再重复拼接，
    否则会得到「线性代数。线性代数。矩阵…」这种重复文本，污染编码输入。
    """
    title = (resource.metadata or {}).get("title") or ""
    body = getattr(resource, "description", "") or ""
    if not body:
        return title
    if not title or body.startswith(title):
        return body
    return f"{title}。{body}"


def resource_texts(bundle) -> list[tuple[int, str]]:
    """(resource_id, 文本) 列表，按 resource_id 升序，保证与向量行号可对齐。"""
    return [(r.resource_id, resource_text(r)) for r in sorted(bundle.resources,
                                                             key=lambda x: x.resource_id)]


def user_need_text(bundle, user_id: int, max_items: int = 20) -> str:
    """学生需求文本 = 该用户最近交互过的若干资源文本的拼接。

    这是「从行为历史反推需求」（平台没有查询/目标字段）。按时间排序取最近的，
    因为越近的行为越能代表当前需求；返回空串表示该用户没有可用文本（冷启动）。
    """
    texts = {r.resource_id: resource_text(r) for r in bundle.resources}
    acts = [b for b in bundle.behaviors if b.user_id == user_id]
    if not acts:
        return ""
    acts.sort(key=lambda x: (x.ts, x.resource_id))
    picked: list[str] = []
    seen: set[int] = set()
    for b in reversed(acts):
        if b.resource_id in seen or b.resource_id not in texts:
            continue
        seen.add(b.resource_id)
        picked.append(texts[b.resource_id])
        if len(picked) >= max_items:
            break
    return " ".join(reversed(picked))
