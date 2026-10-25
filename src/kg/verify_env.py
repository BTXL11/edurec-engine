"""Verify that every library the EMBEDD-ER code depends on can be imported and used."""
import importlib
import os
import sys
import traceback

# This machine's AppData profile is not writable for this session, so keep
# matplotlib's font cache, NLTK's corpora and IPython's state inside the project.
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_HERE, "env", "matplotlib"))
os.environ.setdefault("NLTK_DATA", os.path.join(_HERE, "env", "nltk_data"))
os.environ.setdefault("IPYTHONDIR", os.path.join(_HERE, "env", "ipython"))
os.environ.setdefault("JUPYTER_DATA_DIR", os.path.join(_HERE, "env", "jupyter"))
os.environ.setdefault("PYTHONNOUSERSITE", "1")

FAILURES = []

TARGETS = [
    # module to import, friendly name
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("sklearn.tree", "sklearn.tree"),
    ("sklearn.svm", "sklearn.svm"),
    ("sklearn.neighbors", "sklearn.neighbors"),
    ("sklearn.naive_bayes", "sklearn.naive_bayes"),
    ("sklearn.feature_extraction.text", "sklearn TF-IDF"),
    ("sklearn.manifold", "sklearn TSNE"),
    ("matplotlib", "matplotlib"),
    ("matplotlib.pyplot", "matplotlib.pyplot"),
    ("gensim", "gensim"),
    ("gensim.models", "gensim.models"),
    ("gensim.models.doc2vec", "gensim Doc2Vec"),
    ("rdflib", "rdflib"),
    ("nltk", "nltk"),
    ("nltk.tokenize", "nltk.tokenize"),
    ("nltk.stem", "nltk.stem"),
    ("nltk.corpus", "nltk.corpus"),
    ("networkx", "networkx"),
    ("pyvis", "pyvis"),
    ("pyvis.network", "pyvis.network"),
    ("jieba", "jieba"),
    ("PyPDF2", "PyPDF2"),
    ("Levenshtein", "python-Levenshtein"),
    ("community", "python-louvain (community)"),
    ("lmdb", "lmdb"),
    ("marisa_trie", "marisa-trie"),
    ("pyarrow", "pyarrow"),
    ("streamlit", "streamlit"),
    ("altair", "altair"),
    ("validators", "validators"),
    ("IPython", "ipython"),
    ("ipykernel", "ipykernel"),
]

# optional deep-learning stack (present once the GPU stack is installed)
OPTIONAL = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("tokenizers", "tokenizers"),
    ("sentencepiece", "sentencepiece"),
    ("simpletransformers", "simpletransformers"),
    ("seqeval", "seqeval"),
    ("tensorboard", "tensorboard"),
    ("wandb", "wandb"),
    ("datasets", "datasets"),
    ("huggingface_hub", "huggingface-hub"),
]

print(f"python: {sys.version}")
print(f"executable: {sys.executable}")
print("-" * 70)

for mod, name in TARGETS:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"OK    {name:32s} {ver}")
    except Exception as exc:
        FAILURES.append((name, mod, exc))
        print(f"FAIL  {name:32s} {type(exc).__name__}: {exc}")

print("-" * 70)
print("functional smoke tests")
print("-" * 70)

def check(label, fn):
    try:
        fn()
        print(f"OK    {label}")
    except Exception as exc:
        FAILURES.append((label, "-", exc))
        print(f"FAIL  {label}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)

import numpy as np


def _numpy_ops():
    a = np.random.rand(4, 4)
    b = np.linalg.norm(a, axis=1)
    c = np.dot(a, a)
    assert b.shape == (4,) and c.shape == (4, 4)


def _sklearn_pipeline():
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.random.rand(40, 6)
    y = np.random.randint(0, 2, 40)
    for clf in (DecisionTreeClassifier(random_state=0),
                make_pipeline(StandardScaler(), SVC(gamma="auto")),
                GaussianNB(),
                KNeighborsClassifier()):
        clf.fit(X, y)
        assert clf.predict(X).shape == (40,)
    gs = GridSearchCV(DecisionTreeClassifier(), {"max_depth": [1, 2]}, cv=2)
    gs.fit(X, y)
    cross_val_score(gs.best_estimator_, X, y, cv=2)


def _tfidf_tsne():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE
    docs = ["machine learning course", "linear algebra lecture",
            "database systems and sql", "computer graphics rendering"]
    X = TfidfVectorizer().fit_transform(docs).toarray()
    assert X.shape[0] == 4
    # sklearn 1.1.x has no max_iter; this mirrors embeddings/tfidf.py's TSNE use
    TSNE(n_components=2, perplexity=2, init="random", learning_rate="auto").fit_transform(X)


def _gensim_doc2vec():
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    docs = [TaggedDocument(words=["a", "b", "c"], tags=[0]),
            TaggedDocument(words=["d", "e", "f"], tags=[1])]
    m = Doc2Vec(docs, vector_size=16, min_count=1, epochs=5)
    assert len(m.dv[0]) == 16


def _gensim_word2vec_format(tmp="w2v_probe.txt"):
    """Reproduce the KeyedVectors.load_word2vec_format call EMBEDD-ER uses."""
    import os
    from gensim.models import KeyedVectors
    with open(tmp, "w", encoding="utf8") as fh:
        fh.write("3 4\n")
        fh.write("ENTITY/Alpha 0.1 0.2 0.3 0.4\n")
        fh.write("ENTITY/Beta 0.2 0.3 0.4 0.5\n")
        fh.write("ENTITY/Gamma 0.3 0.4 0.5 0.6\n")
    kv = KeyedVectors.load_word2vec_format(tmp, binary=False)
    assert kv["ENTITY/Alpha"].shape == (4,)
    os.remove(tmp)


def _rdflib_roundtrip():
    from rdflib import Graph, Literal, Namespace, RDF, URIRef
    g = Graph()
    UN = Namespace("https://univ-nantes.fr/ontology/")
    s = URIRef("http://en.wikipedia.org/wiki/Linear_algebra")
    g.add((s, UN.pageRank, Literal(0.5)))
    g.add((s, RDF.type, URIRef("http://example.org/Concept")))
    ttl = g.serialize(format="turtle")
    g2 = Graph()
    g2.parse(data=ttl, format="turtle")
    assert len(g2) == 2
    # the exact access pattern node_embeddings() relies on: predicate endswith pageRank
    preds = [str(p) for _, p, _ in g2]
    assert any(p.endswith("pageRank") for p in preds), preds
    assert any(p.endswith("type") for p in preds), preds


def _rdflib_parse_file():
    import os
    from rdflib import Graph
    p = "probe_graph.ttl"
    with open(p, "w", encoding="utf8") as fh:
        fh.write('@prefix ex: <http://example.org/> .\nex:A ex:b "c" .\n')
    g = Graph()
    g.parse(p, format="turtle")
    assert len(g) == 1
    os.remove(p)


def _matplotlib_agg():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(np.random.rand(5, 5), cmap="gray", interpolation="nearest")
    fig.savefig("probe_plot.png", format="PNG")
    plt.close(fig)
    import os
    assert os.path.getsize("probe_plot.png") > 0
    os.remove("probe_plot.png")


def _networkx_pyvis():
    import networkx as nx
    from pyvis.network import Network
    gnx = nx.MultiDiGraph(name="probe")
    gnx.add_edge("a", "b", rel="r")
    pos = nx.spring_layout(gnx)
    assert len(pos) == 2
    net = Network("500px", "1800px", directed=True, select_menu=True, filter_menu=True)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", title="r", color="green")
    net.toggle_physics(True)
    net.show_buttons(filter_=["physics"])
    net.save_graph("probe_graph.html")
    import os
    assert os.path.getsize("probe_graph.html") > 0
    os.remove("probe_graph.html")


def _torch_gpu():
    """Exercise the RTX 4060 through torch CUDA kernels (sm_89 needs CUDA>=11.8)."""
    import torch
    print(f"      torch {torch.__version__}, built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False - CPU-only build or driver issue")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"      device: {name} (compute capability {cap[0]}.{cap[1]})")
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    assert c.shape == (512, 512)
    print(f"      fp32 matmul on GPU OK, result checksum {float(c.sum()):.3f}")
    half = (a.half() @ b.half()).float()
    torch.cuda.synchronize()
    assert half.shape == (512, 512)
    print("      fp16 matmul on GPU OK (tensor cores)")


def _simpletransformers_import():
    from simpletransformers.language_representation import RepresentationModel  # noqa: F401


def _nltk():
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords  # noqa: F401
    WordNetLemmatizer()
    word_tokenize("the quick brown fox jumps over the lazy dog")


def _jieba():
    import jieba
    assert len(list(jieba.cut("自然语言处理"))) > 0


check("numpy linear algebra", _numpy_ops)
check("sklearn classifiers + GridSearchCV", _sklearn_pipeline)
check("sklearn TF-IDF + TSNE", _tfidf_tsne)
check("gensim Doc2Vec training", _gensim_doc2vec)
check("gensim KeyedVectors.load_word2vec_format", _gensim_word2vec_format)
check("rdflib graph build/serialize/parse", _rdflib_roundtrip)
check("rdflib parse .ttl file", _rdflib_parse_file)
check("matplotlib PNG rendering (Agg)", _matplotlib_agg)
check("networkx + pyvis graph export", _networkx_pyvis)
check("nltk tokenize/lemmatize", _nltk)
check("jieba Chinese segmentation", _jieba)

print("-" * 70)
print("optional deep-learning stack")
print("-" * 70)
OPTIONAL_FAILURES = []
for mod, name in OPTIONAL:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        if mod == "torch":
            ver = f"{ver} cuda_build={m.version.cuda}"
        print(f"OK    {name:32s} {ver}")
    except Exception as exc:
        OPTIONAL_FAILURES.append((name, exc))
        print(f"MISS  {name:32s} {type(exc).__name__}: {exc}")

if all(m != "torch" for m, _ in OPTIONAL_FAILURES):
    check("torch GPU (CUDA 11.8, sm_89)", _torch_gpu)
if all(m != "simpletransformers" for m, _ in OPTIONAL_FAILURES):
    check("simpletransformers RepresentationModel import", _simpletransformers_import)

print("-" * 70)
if FAILURES:
    print(f"FAILURES: {len(FAILURES)}")
    for name, mod, exc in FAILURES:
        print(f"  - {name} ({mod}): {type(exc).__name__}: {exc}")
    sys.exit(1)
if OPTIONAL_FAILURES:
    print(f"core OK; optional stack missing {len(OPTIONAL_FAILURES)}: "
          + ", ".join(n for n, _ in OPTIONAL_FAILURES))
sys.exit(0)
