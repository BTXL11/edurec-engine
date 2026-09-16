import numpy as np
import pytest

from engine.config import EngineConfig
from engine.data.schema import User, Resource, Behavior, DataBundle
from engine.features.encoder import (
    CachedEncoder, Encoder, LocalTfidfSvdEncoder, SentenceTransformerEncoder,
    build_encoder, l2_normalize, resource_text, resource_texts, text_hash,
    user_need_text,
)


# --- 归一化与哈希 ---

def test_l2_normalize_rows_are_unit_length():
    mat = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(mat)
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], 0.0)          # 零向量不产生 NaN
    assert not np.isnan(out).any()


def test_text_hash_is_deterministic_and_distinguishes():
    assert text_hash("线性代数") == text_hash("线性代数")
    assert text_hash("线性代数") != text_hash("线性代数 ")
    assert len(text_hash("x")) == 32


# --- 本地降级编码器 ---

CORPUS = [
    "线性代数。矩阵与向量空间，特征值是核心概念。",
    "线性代数。矩阵运算、特征值分解与线性变换。",
    "Python 入门。变量、循环与函数定义。",
    "Python 进阶。面向对象编程与装饰器。",
]


def _fitted() -> LocalTfidfSvdEncoder:
    return LocalTfidfSvdEncoder(dim=64, seed=42).fit(CORPUS)


def test_local_encoder_shape_and_normalization():
    enc = _fitted()
    vecs = enc.encode(CORPUS)
    assert vecs.shape == (4, 64)
    assert vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_local_encoder_is_deterministic():
    a = _fitted().encode(CORPUS)
    b = _fitted().encode(CORPUS)
    assert np.allclose(a, b)


def test_local_encoder_same_topic_more_similar_than_cross_topic():
    """同类目课程对的余弦相似度应高于跨类目——这是降级实现的验收门槛。"""
    enc = _fitted()
    v = enc.encode(CORPUS)
    same_topic = float(v[0] @ v[1])          # 两门线性代数
    cross_topic = float(v[0] @ v[2])         # 线性代数 vs Python
    assert same_topic > cross_topic


def test_local_encoder_requires_fit():
    enc = LocalTfidfSvdEncoder(dim=32)
    with pytest.raises(RuntimeError, match="fit"):
        enc.encode(["未拟合"])


def test_local_encoder_handles_empty_input():
    enc = _fitted()
    assert enc.encode([]).shape == (0, 64)
    assert enc.encode([""]).shape == (1, 64)


def test_local_encoder_dims_stable_on_tiny_corpus():
    """输出维度必须恒等于请求的 dim，不随语料规模变化（否则缓存与下游都不可靠）。"""
    enc = LocalTfidfSvdEncoder(dim=384, seed=0).fit(["只有一条语料", "第二条"])
    vecs = enc.encode(["只有一条语料"])
    assert vecs.shape == (1, 384)
    assert enc.dim == 384


def test_local_encoder_single_text_corpus_is_usable():
    enc = LocalTfidfSvdEncoder(dim=32, seed=0).fit(["唯一一条语料"])
    assert enc.encode(["唯一一条语料", "另一条"]).shape == (2, 32)


def test_version_includes_corpus_fingerprint():
    """语料变了，词权重就变了：version 必须跟着变，缓存才能自动失效。"""
    a = LocalTfidfSvdEncoder(dim=16, seed=0).fit(CORPUS)
    b = LocalTfidfSvdEncoder(dim=16, seed=0).fit(CORPUS + ["新增语料"])
    assert "corpus=" in a.version
    assert a.version != b.version
    # 同一语料（顺序不同）指纹相同 → 缓存可复用
    c = LocalTfidfSvdEncoder(dim=16, seed=0).fit(list(reversed(CORPUS)))
    assert a.version == c.version


# --- 缓存 ---

def test_cached_encoder_reuses_disk_cache(tmp_path):
    cache_path = str(tmp_path / "cache.npz")
    first = CachedEncoder(_fitted(), cache_path)
    v1 = first.encode(CORPUS)
    assert (tmp_path / "cache.npz").is_file()

    # 新的内部编码器（未拟合也能命中缓存）不应重新计算，结果一致
    second = CachedEncoder(LocalTfidfSvdEncoder(dim=64, seed=42), cache_path)
    v2 = second.encode(CORPUS)
    assert np.allclose(v1, v2)


def test_cached_encoder_only_encodes_misses(tmp_path):
    inner = _fitted()

    class Counting(LocalTfidfSvdEncoder):
        calls = 0

        def encode(self, texts):
            Counting.calls += 1
            return super().encode(texts)

    enc = CachedEncoder(Counting(dim=64, seed=42).fit(CORPUS), str(tmp_path / "c.npz"))
    enc.encode(CORPUS)
    n_after_first = Counting.calls
    enc.encode(CORPUS)                        # 全部命中
    assert Counting.calls == n_after_first
    enc.encode(["一条全新的文本"])              # 只有 miss 才回落到内部编码器
    assert Counting.calls == n_after_first + 1


def test_cache_version_separates_files(tmp_path):
    a = CachedEncoder(LocalTfidfSvdEncoder(dim=32, seed=1), str(tmp_path / "a.npz"))
    b = CachedEncoder(LocalTfidfSvdEncoder(dim=64, seed=1), str(tmp_path / "b.npz"))
    assert a.version != b.version
    assert a.version.startswith("cached(")


# --- 工厂与占位实现 ---

def test_build_encoder_local_with_cache(tmp_path):
    cfg = EngineConfig(encoder_kind="local", encoder_dim=32, model_dir=str(tmp_path))
    enc = build_encoder(cfg)
    assert isinstance(enc, CachedEncoder)
    assert isinstance(enc.inner, LocalTfidfSvdEncoder)
    assert enc.inner.dim == 32
    enc.fit(CORPUS)
    assert enc.encode(CORPUS).shape == (4, 32)


def test_build_encoder_without_cache():
    cfg = EngineConfig(encoder_kind="local", encoder_cache=False, encoder_dim=16)
    enc = build_encoder(cfg)
    assert isinstance(enc, LocalTfidfSvdEncoder)


def test_build_encoder_rejects_unknown_kind():
    with pytest.raises(ValueError, match="encoder_kind"):
        build_encoder(EngineConfig(encoder_kind="magic"))


def test_sentence_transformer_encoder_reports_missing_dependency():
    """当前环境没有 sentence-transformers/网络：应给出可操作的报错而非崩在 import。"""
    enc = SentenceTransformerEncoder()
    with pytest.raises(RuntimeError) as exc:
        enc.encode(["任意文本"])
    assert "sentence-transformers" in str(exc.value)


def test_encoder_is_abstract():
    with pytest.raises(TypeError):
        Encoder()                              # 抽象类不可直接实例化


# --- 文本构造 ---

def _bundle() -> DataBundle:
    resources = [
        Resource(resource_id=1, type="course", category_id=0, tags=(),
                 metadata={"title": "线性代数"}, description="矩阵与特征值"),
        Resource(resource_id=2, type="course", category_id=1, tags=(),
                 metadata={}, description="只有正文没有标题"),
        Resource(resource_id=3, type="course", category_id=1, tags=(),
                 metadata={"title": "只有标题"}, description=""),
    ]
    users = [User(user_id=7), User(user_id=8)]
    behaviors = [
        Behavior(user_id=7, resource_id=1, action="view", ts=100),
        Behavior(user_id=7, resource_id=2, action="view", ts=200),
        Behavior(user_id=7, resource_id=1, action="view", ts=300),   # 重复交互
        Behavior(user_id=7, resource_id=3, action="view", ts=400),
    ]
    return DataBundle(users=users, resources=resources, behaviors=behaviors, ratings=[])


def test_resource_text_combines_title_and_body():
    b = _bundle()
    by_id = {r.resource_id: r for r in b.resources}
    assert resource_text(by_id[1]) == "线性代数。矩阵与特征值"
    assert resource_text(by_id[2]) == "只有正文没有标题"     # 缺标题时退回正文
    assert resource_text(by_id[3]) == "只有标题"             # 缺正文时退回标题


def test_resource_texts_is_sorted_by_id():
    items = resource_texts(_bundle())
    assert [i for i, _ in items] == [1, 2, 3]


def test_user_need_text_uses_recent_unique_items():
    text = user_need_text(_bundle(), user_id=7)
    # 去重后按时间倒序取最近 3 门，再正序拼接
    assert "只有标题" in text
    assert "只有正文没有标题" in text
    assert "线性代数" in text
    assert text.count("线性代数。矩阵与特征值") == 1          # 重复交互只算一次


def test_user_need_text_empty_for_cold_user():
    assert user_need_text(_bundle(), user_id=8) == ""
    assert user_need_text(_bundle(), user_id=999) == ""
