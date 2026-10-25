"""课程知识图谱可视化（适配 数学建模.json -> output/graphs/v01/*.ttl）。

设计说明
--------
当前数据是 `data/数学建模.json` 中的 55 条视频资源（B 站数学建模教程），
每条经过 `run_batch_B.py` 处理后生成 `output/graphs/v01/<Cid>.ttl`。

Cid 定义：`run_batch_B.py` 里 `Cid = safe_filename(str(id))`，由于 `id` 是整数，
因此 `Cid == str(id)`，`<Cid>.ttl` 文件名去掉 `.ttl` 就是 Cid。

课程向量：原脚本依赖 MOOCCube 的预训练 embedding，本项目没有。改为
基于 Wikifier 概念 pageRank 的 TF-IDF 加权向量（与 tfidf_scores 一致），
再算余弦相似度。同数据、零依赖，且天然反映"哪些视频讲到同一批概念"。

产出（output/plots/v01/）：
    01_资源相似度网络.html     资源按概念 TF-IDF 余弦连边
    02_资源概念指纹.html       每条资源一组 TF-IDF 最高的概念
    03_概念资源热力图.png      概念 × 资源 的覆盖热力图
    04_单资源图谱/            每条资源一张概念图（TF-IDF 加权，需 --single）
    00_总览.png                静态总览
"""
import glob
import json
import math
import os
import re
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rdflib  # noqa: E402
from pyvis.network import Network  # noqa: E402

from embeddings.embedder import decode_concept  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
GRAPH_DIR = os.path.join(ROOT, "output", "graphs", "v02")
OUT_DIR = os.path.join(ROOT, "output", "plots", "v02")
DATA_PATH = os.path.join(ROOT, "data", "数学建模.json")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# --------------------------------------------------------------------------- #
# 概念过滤：只去标点/虚词/纯符号；误链类噪声靠 TF-IDF 压制（规则判定不了）
# --------------------------------------------------------------------------- #
PUNCT_NAMES = {
    "逗號", "逗号", "句号", "句號", "问号", "問號", "感叹号", "感嘆號", "省略号", "省略號",
    "分号", "分號", "冒号", "冒號", "括号", "括號", "引号", "引號", "书名号", "書名號",
    "破折号", "破折號", "连接号", "連接號", "斜線", "斜线", "反斜線", "反斜杠", "空格",
    "分隔号", "分隔號", "顿号", "頓號", "星号", "星號", "井號", "井号", "下划线", "底線",
    "逻辑与", "邏輯與", "逻辑或", "邏輯或", "逻辑非", "不等號", "不等号", "等号", "等號",
    "大于号", "小於號", "加号", "減號", "乘號", "除號", "百分号", "百分號", "波浪號",
}
STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "你", "我", "他", "它", "这", "那", "有",
    "用部", "語氣", "语气", "用", "以", "及", "等", "為", "为", "被", "把", "則", "则",
    "讓", "让", "使", "對", "对", "從", "从", "到", "而", "並", "并", "但", "若", "如",
    "衝繁疲難", "冲繁疲难", "系詞", "系词", "了別",
}


def is_junk(concept, min_len=2):
    c = concept.strip()
    if len(c) < min_len:
        return True
    if c in PUNCT_NAMES or c in STOPWORDS:
        return True
    if re.fullmatch(r"[\W\d_]+", c):
        return True
    if re.fullmatch(r"[A-Za-z]", c):
        return True
    m = re.match(r"^(.+?)\s*[（(]", c)
    if m and len(m.group(1).strip()) < 2:
        return True
    return False


# --------------------------------------------------------------------------- #
# 资源信息加载（替代 MOOCCube 的 load_courses / build_reverse_map）
# --------------------------------------------------------------------------- #
def load_resource_info():
    """从 数学建模.json 读取 {id_str -> {title, author, type}}。"""
    with open(DATA_PATH, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        rows = data.get("rows") or []
    else:
        rows = data
    info = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id", "")).strip()
        if not rid:
            continue
        info[rid] = {
            "title": (r.get("title") or "").strip(),
            "author": (r.get("author") or "").strip(),
            "type": (r.get("type") or "").strip(),
        }
    return info


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def save_net(net, abs_path):
    """pyvis 会把路径拼成 './<path>'，传绝对路径在 Windows 上会报 Invalid argument。"""
    net.save_graph(os.path.relpath(abs_path, ROOT))
    return abs_path


def short_name(title, maxlen=22):
    """标题太长，截断给可视化用。"""
    s = (title or "").strip()
    if not s:
        return "?"
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


def build_name_map(course_graphs, resource_info):
    """Cid -> 显示名（标题截断），重名加 ①②③ 消歧。

    返回 (label: cid->显示名, raw_of: cid->原始 id 字符串)。
    """
    from collections import Counter

    labels = {}
    raw_of = {}
    for cid in course_graphs:
        raw_of[cid] = cid                       # 文件名即 id
        info = resource_info.get(cid) or {}
        title = info.get("title") or ""
        labels[cid] = short_name(title) if title else cid

    counts = Counter(labels.values())
    seen = {}
    for cid in sorted(labels):
        nm = labels[cid]
        if counts[nm] > 1:
            seen[nm] = seen.get(nm, 0) + 1
            idx = seen[nm]
            mark = "①②③④⑤⑥⑦⑧⑨"[idx - 1] if idx <= 9 else f"({idx})"
            labels[cid] = f"{nm}{mark}"
    return labels, raw_of


def fit_label(text, maxlen):
    return text if len(text) <= maxlen else text[: maxlen - 1] + "…"


def load_course_graphs(verbose=True):
    out = {}
    for path in sorted(glob.glob(os.path.join(GRAPH_DIR, "*.ttl"))):
        cid = os.path.basename(path)[:-4]        # 去 .ttl 后缀即为 Cid
        g = rdflib.Graph()
        g.parse(path, format="turtle")
        concepts = {}
        for s, p, o in g:
            if "pageRank" in str(p):
                name = decode_concept(s)
                if not is_junk(name):
                    concepts[name] = float(o)
        out[cid] = {"graph": g, "concepts": concepts}
    if verbose:
        tot = sum(len(v["concepts"]) for v in out.values())
        med = int(np.median([len(v["concepts"]) for v in out.values()])) if out else 0
        print(f"读入 {len(out)} 条资源图，过滤后概念合计 {tot}，中位 {med}")
    return out


def tfidf_scores(course_graphs):
    """返回 {cid: [(concept, tfidf, pageRank), ...]} 与 df 表。"""
    n_doc = len(course_graphs)
    df = {}
    for info in course_graphs.values():
        for c in info["concepts"]:
            df[c] = df.get(c, 0) + 1

    out = {}
    for cid, info in course_graphs.items():
        rows = []
        for c, pr in info["concepts"].items():
            idf = math.log(n_doc / df[c]) if df[c] else 0.0
            rows.append((c, pr * idf, pr))
        rows.sort(key=lambda t: -t[1])
        out[cid] = rows
    return out, df


# --------------------------------------------------------------------------- #
# 相似度向量：用概念 TF-IDF 拼接课程向量（替代 MOOCCube embedding）
# --------------------------------------------------------------------------- #
def build_tfidf_vectors(scores, df):
    """每门课 -> 归一化的概念 TF-IDF 向量。"""
    vocab = sorted(df)
    cidx = {c: i for i, c in enumerate(vocab)}
    vecs = {}
    for cid, rows in scores.items():
        v = np.zeros(len(vocab), dtype=np.float64)
        for c, s, _ in rows:
            if c in cidx and s > 0:
                v[cidx[c]] = s
        n = np.linalg.norm(v)
        if n > 0:
            vecs[cid] = v / n
    return vecs


# --------------------------------------------------------------------------- #
# ① 资源相似度网络
# --------------------------------------------------------------------------- #
def plot_course_similarity(course_graphs, scores, df, out_dir,
                           labels=None, raw_of=None, threshold=None):
    labels = labels or {}
    raw_of = raw_of or {}
    vecs = build_tfidf_vectors(scores, df)
    if len(vecs) < 2:
        print("  向量不足，跳过")
        return None, None

    ids = sorted(vecs)
    M = np.vstack([vecs[c] for c in ids])
    S = M @ M.T

    if threshold is None:
        triu = S[np.triu_indices_from(S, k=1)]
        nz = triu[triu > 0]
        threshold = float(np.percentile(nz, 25)) if len(nz) else 0.05
        threshold = max(0.03, threshold)

    # 打印分布，方便调试阈值
    triu = S[np.triu_indices_from(S, k=1)]
    nz = triu[triu > 0]
    if len(nz):
        print(f"  相似度分布：非零边 {len(nz)}，"
              f"min={nz.min():.3f} p25={np.percentile(nz,25):.3f} "
              f"p50={np.percentile(nz,50):.3f} max={nz.max():.3f}")
    print(f"  连边阈值 = {threshold:.3f}")

    net = Network(height="1000px", width="100%", bgcolor="#ffffff", font_color="#222222",
                  select_menu=True, cdn_resources="in_line",
                  heading=f"EMBEDD-ER 资源相似度网络（概念 TF-IDF 余弦 ≥ {threshold:.3f}）")
    for cid in ids:
        nm = labels.get(cid, cid)
        net.add_node(cid, label=nm, shape="dot", size=18,
                     color="#ffb347", course=nm,
                     title=f"{nm}<br>Cid={raw_of.get(cid, cid)}<br>"
                           f"概念数 {len(course_graphs[cid]['concepts'])}")
    n = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if S[i, j] >= threshold:
                w = float(S[i, j])
                net.add_edge(ids[i], ids[j], value=w, title=f"余弦 {w:.3f}",
                             width=0.6 + (w - threshold) / max(1e-9, 1 - threshold) * 5,
                             color="#3b7dd8")
                n += 1
    net.toggle_physics(True)
    net.show_buttons(filter_=["physics"])
    save_net(net, os.path.join(out_dir, "01_资源相似度网络.html"))
    print(f"  ① 相似度网络: {len(ids)} 节点 / {n} 边")
    return S, ids


# --------------------------------------------------------------------------- #
# ② 资源概念指纹
# --------------------------------------------------------------------------- #
def plot_fingerprints(scores, df, out_dir, labels=None, raw_of=None, top_k=12):
    labels = labels or {}
    raw_of = raw_of or {}
    n_doc = len(scores)
    net = Network(height="1100px", width="100%", bgcolor="#ffffff", font_color="#222222",
                  select_menu=True, cdn_resources="in_line",
                  heading="每条资源的 TF-IDF 概念指纹（pageRank × log(N/df)，已剔除标点与虚词）")
    for cid in sorted(scores):
        nm = labels.get(cid, cid)
        net.add_node("C:" + cid, label=nm, shape="box", color="#ffb347",
                     size=22, course=nm, title=f"{nm}<br>Cid={raw_of.get(cid, cid)}")
    for cid in sorted(scores):
        hi = scores[cid][0][1] if scores[cid] else 1.0
        for c, score, pr in scores[cid][:top_k]:
            nid = "K:" + c
            share = df.get(c, 1)
            net.add_node(nid, label=c, shape="dot",
                         color="#7fb3d5" if share <= 3 else "#c9d6df",
                         size=8 + (score / max(1e-9, hi)) * 16,
                         title=f"{c}<br>被 {share}/{n_doc} 条资源覆盖<br>"
                               f"TF-IDF {score:.3e}<br>pageRank {pr:.3e}",
                         course="概念")
            net.add_edge("C:" + cid, nid, title=f"TF-IDF {score:.3e}",
                         color="#c8c8c8", width=0.5 + score / max(1e-9, hi) * 3)
    net.toggle_physics(True)
    net.show_buttons(filter_=["physics"])
    save_net(net, os.path.join(out_dir, "02_资源概念指纹.html"))
    total = sum(min(top_k, len(v)) for v in scores.values())
    print(f"  ② 概念指纹: {len(scores)} 资源 / {total} 概念边")


# --------------------------------------------------------------------------- #
# ③ 概念 × 资源 热力图
# --------------------------------------------------------------------------- #
def plot_heatmap(scores, df, out_dir, labels=None, n_concepts=30, per_course=40,
                 max_label=22, max_courses=120):
    labels = labels or {}
    n_doc = len(scores)
    per_course_map = {cid: {c: s for c, s, _ in scores[cid][:per_course]}
                      for cid in scores}

    # 候选：被 >=2 条覆盖、且不是全覆盖的共享概念
    cand = [c for c in df if 2 <= df[c] < n_doc]
    cand.sort(key=lambda c: -max((m.get(c, 0.0) for m in per_course_map.values()),
                                 default=0.0))
    keep = cand[:n_concepts]
    if not keep:
        print("  无可共享概念，跳过热力图")
        return []

    all_ids = sorted(scores)
    subset = False
    if len(all_ids) > max_courses:
        all_ids = sorted(all_ids, key=lambda c: -len(per_course_map.get(c, {})))[:max_courses]
        all_ids = sorted(all_ids)
        subset = True

    mat = np.zeros((len(keep), len(all_ids)))
    for j, cid in enumerate(all_ids):
        for i, c in enumerate(keep):
            mat[i, j] = per_course_map[cid].get(c, 0.0)

    transpose = len(all_ids) > 60
    plot_mat = mat.T if transpose else mat
    note = f"（仅取概念最丰富的前 {len(all_ids)} 条资源）" if subset else ""

    if transpose:
        fig_h = min(max(8.0, len(all_ids) * 0.145), 24.0)
        fig, ax = plt.subplots(figsize=(13, fig_h))
        im = ax.imshow(plot_mat, cmap="YlOrRd", aspect="auto")
        ax.set_yticks(range(len(all_ids)))
        ax.set_yticklabels([fit_label(labels.get(c, c), 40) for c in all_ids], fontsize=6)
        ax.set_xticks(range(len(keep)))
        ax.set_xticklabels([f"{c} ({df[c]})" for c in keep], rotation=90, fontsize=7)
        ax.set_xlabel("概念（括号内为覆盖资源数）")
        ax.set_ylabel("资源")
        ax.set_title(f"资源 × 概念 覆盖热力图（颜色 = TF-IDF；共享概念前 {len(keep)} 个）"
                     f"{note}", fontsize=12)
    else:
        fig, ax = plt.subplots(figsize=(13, max(6, len(keep) * 0.34)))
        im = ax.imshow(plot_mat, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(all_ids)))
        ax.set_xticklabels([fit_label(labels.get(c, c), max_label)
                            for c in all_ids], rotation=90, fontsize=6)
        ax.set_yticks(range(len(keep)))
        ax.set_yticklabels([f"{c}  ({df[c]})" for c in keep], fontsize=8)
        ax.set_title(f"概念 × 资源 覆盖热力图（颜色 = TF-IDF；共享概念前 {len(keep)} 个）"
                     f"{note}", fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.03, label="TF-IDF")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_概念资源热力图.png"), dpi=130)
    plt.close(fig)
    print(f"  ③ 概念热力图: {len(keep)} 概念 × {len(all_ids)} 资源"
          + ("（已转置）" if transpose else "") + ("（取子集）" if subset else ""))
    return keep


# --------------------------------------------------------------------------- #
# ④ 单资源图谱
# --------------------------------------------------------------------------- #
def plot_single_courses(course_graphs, scores, out_dir, top_k=50,
                        only=None, labels=None, raw_of=None):
    labels = labels or {}
    raw_of = raw_of or {}
    if only is None:
        print(f"  ④ 单资源图谱: 已跳过（{len(course_graphs)} 条，需用 --single 开启）")
        return

    sub = os.path.join(out_dir, "04_单资源图谱")
    os.makedirs(sub, exist_ok=True)
    targets = [c for c in sorted(course_graphs) if not only or c in only]
    for cid in targets:
        rows = scores[cid][:top_k]
        if not rows:
            continue
        nm = labels.get(cid, cid)
        net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#222222",
                      directed=True, select_menu=True, cdn_resources="in_line",
                      heading=f"{nm}（TF-IDF 前 {len(rows)} 个概念）")
        net.add_node("CHAPTER", label=nm, shape="box", color="#ffb347",
                     size=24, course="资源", title=f"{nm}<br>Cid={raw_of.get(cid, cid)}")
        hi = rows[0][1] or 1.0
        for c, score, pr in rows:
            frac = score / hi
            net.add_node(c, label=c, shape="dot", color="#7fb3d5",
                         size=8 + frac * 20, course="概念",
                         title=f"{c}<br>TF-IDF {score:.3e}<br>pageRank {pr:.3e}")
            net.add_edge("CHAPTER", c, title=f"权重 {score:.2e}",
                         color="#d0d0d0", width=0.4 + frac * 2.5)
        net.toggle_physics(True)
        net.show_buttons(filter_=["physics"])
        save_net(net, os.path.join(sub, f"{cid}.html"))
    print(f"  ④ 单资源图谱: {len(targets)} 张 -> output/plots/v01/04_单资源图谱/")


# --------------------------------------------------------------------------- #
# ⑤ 静态总览
# --------------------------------------------------------------------------- #
def plot_overview(course_graphs, S, ids, df, out_dir, labels=None, max_label=18):
    labels = labels or {}
    n_courses = len(course_graphs)
    show_labels = n_courses <= 60
    fig = plt.figure(figsize=(19, 7.5))

    if S is not None:
        ax1 = fig.add_subplot(1, 3, 1)
        sizes = np.asarray([len(course_graphs[c]["concepts"]) for c in ids])
        order = np.argsort(-sizes)
        S2 = S[np.ix_(order, order)]
        im = ax1.imshow(S2, cmap="viridis", vmin=float(S2.min()), vmax=1.0)
        if show_labels:
            tick_names = [fit_label(labels.get(ids[i], ids[i]), max_label) for i in order]
            ax1.set_xticks(range(len(tick_names)))
            ax1.set_yticks(range(len(tick_names)))
            ax1.set_xticklabels(tick_names, rotation=90, fontsize=5)
            ax1.set_yticklabels(tick_names, fontsize=5)
        else:
            ax1.set_xticks([])
            ax1.set_yticks([])
        ax1.set_title(f"资源概念 TF-IDF 余弦相似度（{len(ids)} 条）", fontsize=12)
        fig.colorbar(im, ax=ax1, fraction=0.046)
        if not show_labels:
            ax1.set_xlabel("资源过多，标签见 01_资源相似度网络.html", fontsize=8)

    ax2 = fig.add_subplot(1, 3, 2)
    dfdist = {}
    for n in df.values():
        dfdist[n] = dfdist.get(n, 0) + 1
    ks = sorted(dfdist)
    step = max(1, len(ks) // 15)
    ax2.bar([str(k) for k in ks], [dfdist[k] for k in ks], color="#5b8ff9")
    ax2.set_yscale("log")
    ax2.set_xticks(range(0, len(ks), step))
    ax2.set_xticklabels([str(ks[i]) for i in range(0, len(ks), step)],
                        rotation=45, fontsize=7)
    ax2.set_xlabel("概念被多少条资源覆盖")
    ax2.set_ylabel("概念数量（对数）")
    ax2.set_title(f"概念文档频率分布（过滤后 {len(df)} 个）", fontsize=12)

    ax3 = fig.add_subplot(1, 3, 3)
    top = sorted(df.items(), key=lambda kv: -kv[1])[:15][::-1]
    ax3.barh([c for c, _ in top], [n for _, n in top], color="#ff9f43")
    ax3.set_xlabel("覆盖资源数")
    ax3.set_title("覆盖资源最多的概念（含误链，是噪声提示）", fontsize=11)
    ax3.tick_params(axis="y", labelsize=8)

    fig.suptitle(f"EMBEDD-ER 知识图谱总览（{n_courses} 条数学建模视频资源）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out_dir, "00_总览.png"), dpi=130)
    plt.close(fig)
    print("  ⑤ 静态总览: 00_总览.png")


def cleanup_pyvis_lib():
    import shutil
    lib = os.path.join(ROOT, "lib")
    if os.path.exists(os.path.join(lib, "vis-9.1.2", "vis-network.min.js")):
        shutil.rmtree(lib, ignore_errors=True)
        print("  已清理 pyvis 生成的 lib/ 目录")


def main():
    # --single           每条都生成一张图
    # --single <Cid>     只为指定 Cid 生成，可重复
    # --threshold 0.08   手动指定相似度连边阈值
    argv = sys.argv[1:]
    single = None
    if "--single" in argv:
        i = argv.index("--single")
        rest = argv[i + 1:]
        wanted = [a for a in rest if not a.startswith("--")]
        single = set(wanted) if wanted else set()

    threshold = None
    if "--threshold" in argv:
        i = argv.index("--threshold")
        try:
            threshold = float(argv[i + 1])
        except (IndexError, ValueError):
            print("--threshold 需要跟一个数字")
            sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    print("加载资源图 ...")
    cg = load_course_graphs()
    if not cg:
        print(f"未在 {GRAPH_DIR} 找到任何 .ttl，先跑 run_batch_B.py")
        return

    print("\n按 TF-IDF 加权 ...")
    scores, df = tfidf_scores(cg)
    print(f"  概念词汇表 {len(df)} 个")

    print("加载资源元信息 ...")
    resource_info = load_resource_info()
    labels, raw_of = build_name_map(cg, resource_info)
    print(f"  资源名映射 {len(labels)} 个（重名已加 ①②③ 消歧）\n")

    print("生成可视化 -> output/plots/v01/\n")
    S, ids = plot_course_similarity(cg, scores, df, OUT_DIR,
                                    labels=labels, raw_of=raw_of,
                                    threshold=threshold)
    plot_fingerprints(scores, df, OUT_DIR, labels=labels, raw_of=raw_of)
    plot_heatmap(scores, df, OUT_DIR, labels=labels)
    plot_single_courses(cg, scores, OUT_DIR, only=single, labels=labels, raw_of=raw_of)
    plot_overview(cg, S, ids, df, OUT_DIR, labels=labels)
    cleanup_pyvis_lib()

    print(f"\n各资源 TF-IDF 最高的 8 个概念（前 15 条，共 {len(scores)} 条）:")
    for cid in sorted(scores)[:15]:
        top = " / ".join(c for c, _, _ in scores[cid][:8])
        print(f"  {short_name(labels.get(cid, cid), 22):24s} {top}")
    if len(scores) > 15:
        print(f"  ...（其余 {len(scores) - 15} 条略）")
    print("\n完成。")


if __name__ == "__main__":
    main()