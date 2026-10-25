"""课程知识图谱可视化（20 门 MOOCCube 课程）。

设计说明
--------
Wikifier 对中文的标注噪声很大：1889 个概念里 60% 只出现在 1 门课，
覆盖全部 20 门课的只有"句号""逗號"两个标点；单字/双字概念占 49%；
"萧何""精液""不丹""习近平"这类与课程无关的条目会以中等文档频率混进来。

因此可视化不用"原始共现次数"，而用 TF-IDF 式加权：

    score(c, course) = pageRank(c, course) × log(N / df(c))

即"在这门课里重要"且"不是所有课都有的泛概念"。这样每门课的 top 概念
才反映它的学科指纹。

产出（output/plots/v01/）：
    01_课程相似度网络.html     课程按余弦相似度连边
    02_课程概念指纹.html       每门课一组 TF-IDF 最高的概念
    03_概念课程热力图.png      概念 × 课程 的覆盖热力图
    04_单课程图谱/            每门课一张概念图（TF-IDF 加权）
    00_总览.png                静态总览
"""
import glob
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
from recommender.build_embeddings import load_embeddings  # noqa: E402
from recommender.idmap import build_reverse_map, cid_from_filename  # noqa: E402
from recommender.stats import load_courses  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)  # 沙箱下进程 CWD 可能落到父目录，统一到项目根
GRAPH_DIR = os.path.join(ROOT, "output", "graphs", "v01")
OUT_DIR = os.path.join(ROOT, "output", "plots", "v01")

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
# 工具
# --------------------------------------------------------------------------- #
def save_net(net, abs_path):
    """pyvis 会把路径拼成 './<path>'，传绝对路径在 Windows 上会报 Invalid argument。"""
    net.save_graph(os.path.relpath(abs_path, ROOT))
    return abs_path


def short_name(cid, maxlen=26):
    """从课程 ID 截出可读短名，仅在取不到课程名时兜底。"""
    s = cid.replace("C_course-v1:", "").replace("C_course-v1_", "")
    for suf in ("+sp", "+2019_T1", "+2018_T2"):
        s = s.replace(suf, "")
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


def build_name_map(course_graphs):
    """课程 ID -> 课程名，并处理重名。

    course.json 的 name 才是真正的课程名（如"初级财务会计（自主模式）"），
    比从 ID 截出的 "ACCA+FA1_X" 可读得多。有 6 组课程重名（例如 3 门
    "中国近现代史纲要（自主模式）"），加 ①②③ 标记消歧，否则图上分不清谁是谁。
    返回 (label: safe_cid->显示名, raw_of: safe_cid->原始 id)。
    """
    from collections import Counter

    names = {}
    for c in load_courses():
        cid = c.get("id")
        nm = (c.get("name") or "").strip()
        if cid and nm and cid not in names:
            names[cid] = nm

    reverse = build_reverse_map()
    raw_of = {safe: (reverse.get(safe) or safe) for safe in course_graphs}

    label = {safe: (names.get(raw) or short_name(safe)) for safe, raw in raw_of.items()}

    counts = Counter(label.values())
    seen = {}
    for safe in sorted(label):
        nm = label[safe]
        if counts[nm] > 1:
            seen[nm] = seen.get(nm, 0) + 1
            idx = seen[nm]
            mark = "①②③④⑤⑥⑦⑧⑨"[idx - 1] if idx <= 9 else f"({idx})"
            label[safe] = f"{nm}{mark}"
    return label, raw_of


def fit_label(text, maxlen):
    """静态图（matplotlib）的坐标轴标签必须限长，否则 582 门课会挤成一片。"""
    return text if len(text) <= maxlen else text[: maxlen - 1] + "…"


def load_course_graphs(verbose=True):
    out = {}
    for path in sorted(glob.glob(os.path.join(GRAPH_DIR, "*.ttl"))):
        cid = cid_from_filename(path)
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
        print(f"读入 {len(out)} 门课程图，过滤后概念合计 {tot}，"
              f"中位 {int(np.median([len(v['concepts']) for v in out.values()]))}")
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
# ① 课程相似度网络
# --------------------------------------------------------------------------- #
def plot_course_similarity(course_graphs, out_dir, labels=None, raw_of=None,
                           threshold=0.95):
    labels = labels or {}
    raw_of = raw_of or {}
    emb, _ = load_embeddings()
    vecs = {cid: emb[cid] for cid in course_graphs
            if cid in emb and np.linalg.norm(emb[cid]) > 0}
    if len(vecs) < 2:
        print("  向量不足，跳过")
        return None, None

    ids = sorted(vecs)
    M = np.vstack([vecs[c] for c in ids])
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    S = Mn @ Mn.T

    net = Network(height="1000px", width="100%", bgcolor="#ffffff", font_color="#222222",
                  select_menu=True, cdn_resources="in_line",
                  heading=f"EMBEDD-ER 课程相似度网络（余弦 ≥ {threshold} 才连边）")
    for cid in ids:
        nm = labels.get(cid, short_name(cid))
        net.add_node(cid, label=nm, shape="dot", size=18,
                     color="#ffb347", course=nm,
                     title=f"{nm}<br>{raw_of.get(cid, cid)}<br>"
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
    save_net(net, os.path.join(out_dir, "01_课程相似度网络.html"))
    print(f"  ① 相似度网络: {len(ids)} 节点 / {n} 边")
    return S, ids


# --------------------------------------------------------------------------- #
# ② 课程概念指纹
# --------------------------------------------------------------------------- #
def plot_fingerprints(scores, df, out_dir, labels=None, raw_of=None, top_k=12):
    labels = labels or {}
    raw_of = raw_of or {}
    n_doc = len(scores)
    net = Network(height="1100px", width="100%", bgcolor="#ffffff", font_color="#222222",
                  select_menu=True, cdn_resources="in_line",
                  heading="每门课的 TF-IDF 概念指纹（pageRank × log(N/df)，已剔除标点与虚词）")
    for cid in sorted(scores):
        nm = labels.get(cid, short_name(cid))
        net.add_node("C:" + cid, label=nm, shape="box", color="#ffb347",
                     size=22, course=nm, title=f"{nm}<br>{raw_of.get(cid, cid)}")
    for cid in sorted(scores):
        hi = scores[cid][0][1] if scores[cid] else 1.0
        for c, score, pr in scores[cid][:top_k]:
            nid = "K:" + c
            share = df.get(c, 1)
            net.add_node(nid, label=c, shape="dot",
                         color="#7fb3d5" if share <= 3 else "#c9d6df",
                         size=8 + (score / max(1e-9, hi)) * 16,
                         title=f"{c}<br>被 {share}/{n_doc} 门课覆盖<br>"
                               f"TF-IDF {score:.3e}<br>pageRank {pr:.3e}",
                         course="概念")
            net.add_edge("C:" + cid, nid, title=f"TF-IDF {score:.3e}",
                         color="#c8c8c8", width=0.5 + score / max(1e-9, hi) * 3)
    net.toggle_physics(True)
    net.show_buttons(filter_=["physics"])
    save_net(net, os.path.join(out_dir, "02_课程概念指纹.html"))
    total = sum(min(top_k, len(v)) for v in scores.values())
    print(f"  ② 概念指纹: {len(scores)} 课程 / {total} 概念边")


# --------------------------------------------------------------------------- #
# ③ 概念 × 课程 热力图
# --------------------------------------------------------------------------- #
def plot_heatmap(scores, df, out_dir, labels=None, n_concepts=30, per_course=40,
                 max_label=22, max_courses=120):
    """概念 × 课程 的 TF-IDF 覆盖矩阵。

    582 门课全画进一张静态图是读不出来的（矩阵被压成细条，标签也糊），
    所以课程多时取"概念数最多"的前 max_courses 门。要看全部课程请用
    01_课程相似度网络.html / 02_课程概念指纹.html，那两个是交互式的。
    """
    labels = labels or {}
    n_doc = len(scores)
    per_course_map = {cid: {c: s for c, s, _ in scores[cid][:per_course]}
                      for cid in scores}

    # 候选：被 >=2 门课覆盖（共享概念）。
    # 排序用"最大 TF-IDF"而不是"TF-IDF 总和"：总和会让出现在很多门课 top-40 里的
    # 泛概念（英语/心理学/清华大学…）胜出 —— 它们在每门课里权重不高，但加起来很大。
    # 最大值选出的才是"在某门课里特别突出"的概念。
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
        # 取概念数最多的若干门；越多概念越能体现矩阵结构
        all_ids = sorted(all_ids, key=lambda c: -len(per_course_map.get(c, {})))[:max_courses]
        all_ids = sorted(all_ids)
        subset = True

    mat = np.zeros((len(keep), len(all_ids)))
    for j, cid in enumerate(all_ids):
        for i, c in enumerate(keep):
            mat[i, j] = per_course_map[cid].get(c, 0.0)

    # 课程一多，x 轴标签就放不下，转置成"课程在 y 轴、标签横向排"
    transpose = len(all_ids) > 60
    plot_mat = mat.T if transpose else mat
    note = f"（仅取概念最丰富的前 {len(all_ids)} 门课程）" if subset else ""

    if transpose:
        fig_h = min(max(8.0, len(all_ids) * 0.145), 24.0)
        fig, ax = plt.subplots(figsize=(13, fig_h))
        im = ax.imshow(plot_mat, cmap="YlOrRd", aspect="auto")
        ax.set_yticks(range(len(all_ids)))
        ax.set_yticklabels([fit_label(labels.get(c, short_name(c)), 40) for c in all_ids],
                           fontsize=6)
        ax.set_xticks(range(len(keep)))
        ax.set_xticklabels([f"{c} ({df[c]}门)" for c in keep], rotation=90, fontsize=7)
        ax.set_xlabel("概念（括号内为覆盖课程数）")
        ax.set_ylabel("课程")
        ax.set_title(f"课程 × 概念 覆盖热力图（颜色 = TF-IDF；共享概念前 {len(keep)} 个）"
                     f"{note}", fontsize=12)
    else:
        fig, ax = plt.subplots(figsize=(13, max(6, len(keep) * 0.34)))
        im = ax.imshow(plot_mat, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(all_ids)))
        ax.set_xticklabels([fit_label(labels.get(c, short_name(c)), max_label)
                            for c in all_ids], rotation=90, fontsize=6)
        ax.set_yticks(range(len(keep)))
        ax.set_yticklabels([f"{c}  ({df[c]}门)" for c in keep], fontsize=8)
        ax.set_title(f"概念 × 课程 覆盖热力图（颜色 = TF-IDF；共享概念前 {len(keep)} 个）"
                     f"{note}", fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.03, label="TF-IDF")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_概念课程热力图.png"), dpi=130)
    plt.close(fig)
    print(f"  ③ 概念热力图: {len(keep)} 概念 × {len(all_ids)} 课程"
          + ("（已转置）" if transpose else "") + ("（取子集）" if subset else ""))
    return keep


# --------------------------------------------------------------------------- #
# ④ 单课程图谱
# --------------------------------------------------------------------------- #
def plot_single_courses(course_graphs, scores, out_dir, top_k=50,
                        only=None, labels=None, raw_of=None):
    """每门课一张图。

    582 门课时这一项会产出约 434 MB / 582 个 HTML，比其余全部产出加起来还大得多，
    所以默认不生成；需要时用 `--single` 显式开启，或用 `--single 课程ID` 只出指定课程。
    """
    labels = labels or {}
    raw_of = raw_of or {}
    if only is None:
        print(f"  ④ 单课程图谱: 已跳过（{len(course_graphs)} 门课，需用 --single 开启）")
        return

    sub = os.path.join(out_dir, "04_单课程图谱")
    os.makedirs(sub, exist_ok=True)
    targets = [c for c in sorted(course_graphs) if not only or c in only]
    for cid in targets:
        rows = scores[cid][:top_k]
        if not rows:
            continue
        nm = labels.get(cid, short_name(cid))
        net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#222222",
                      directed=True, select_menu=True, cdn_resources="in_line",
                      heading=f"{nm}（TF-IDF 前 {len(rows)} 个概念）")
        net.add_node("CHAPTER", label=nm, shape="box", color="#ffb347",
                     size=24, course="课程", title=f"{nm}<br>{raw_of.get(cid, cid)}")
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
    print(f"  ④ 单课程图谱: {len(targets)} 张 -> output/plots/v01/04_单课程图谱/")


# --------------------------------------------------------------------------- #
# ⑤ 静态总览
# --------------------------------------------------------------------------- #
def plot_overview(course_graphs, S, ids, df, out_dir, labels=None, max_label=None):
    """三合一静态总览。

    课程数少时给相似度矩阵标上全部课程名；课程数多到几百门时刻度标签必然糊成一片，
    此时改为不标标签、只说明规模 —— HTML 版（01/02）保留完整课程名可交互查看。
    """
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
            tick_names = [fit_label(labels.get(ids[i], short_name(ids[i])), max_label or 26)
                          for i in order]
            ax1.set_xticks(range(len(tick_names)))
            ax1.set_yticks(range(len(tick_names)))
            ax1.set_xticklabels(tick_names, rotation=90, fontsize=5)
            ax1.set_yticklabels(tick_names, fontsize=5)
        else:
            ax1.set_xticks([])
            ax1.set_yticks([])
        ax1.set_title(f"课程余弦相似度（EMBEDD-ER，{len(ids)} 门）", fontsize=12)
        fig.colorbar(im, ax=ax1, fraction=0.046)
        if not show_labels:
            ax1.set_xlabel("课程数目过多，标签见 01_课程相似度网络.html（鼠标悬停可看课程名）",
                           fontsize=8)

    ax2 = fig.add_subplot(1, 3, 2)
    dfdist = {}
    for n in df.values():
        dfdist[n] = dfdist.get(n, 0) + 1
    ks = sorted(dfdist)
    # 582 门课时 ks 有 582 个取值，全标出来会糊成一片，所以只标稀疏的刻度
    step = max(1, len(ks) // 15)
    ax2.bar([str(k) for k in ks], [dfdist[k] for k in ks], color="#5b8ff9")
    ax2.set_yscale("log")
    ax2.set_xticks(range(0, len(ks), step))
    ax2.set_xticklabels([str(ks[i]) for i in range(0, len(ks), step)],
                        rotation=45, fontsize=7)
    ax2.set_xlabel("概念被多少门课程覆盖")
    ax2.set_ylabel("概念数量（对数）")
    ax2.set_title(f"概念文档频率分布（过滤后 {len(df)} 个）", fontsize=12)

    ax3 = fig.add_subplot(1, 3, 3)
    top = sorted(df.items(), key=lambda kv: -kv[1])[:15][::-1]
    ax3.barh([c for c, _ in top], [n for _, n in top], color="#ff9f43")
    ax3.set_xlabel("覆盖课程数")
    ax3.set_title("覆盖课程最多的概念（含误链，是噪声提示）", fontsize=11)
    ax3.tick_params(axis="y", labelsize=8)

    fig.suptitle(f"EMBEDD-ER 知识图谱总览（{n_courses} 门 MOOCCube 课程）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out_dir, "00_总览.png"), dpi=130)
    plt.close(fig)
    print("  ⑤ 静态总览: 00_总览.png")


def cleanup_pyvis_lib():
    """pyvis 即使以内联方式渲染也会在 CWD 生成 lib/ 目录（约 1.8 MB），
    而生成的 HTML 并没有引用它，所以收尾时删掉，避免污染仓库。"""
    import shutil

    lib = os.path.join(ROOT, "lib")
    if os.path.exists(os.path.join(lib, "vis-9.1.2", "vis-network.min.js")):
        shutil.rmtree(lib, ignore_errors=True)
        print("  已清理 pyvis 生成的 lib/ 目录")


def main():
    # --single           每门课都生成一张图（582 门课约 434 MB，默认关闭）
    # --single <课程ID>  只为指定课程生成，可重复
    argv = sys.argv[1:]
    single = None
    if "--single" in argv:
        i = argv.index("--single")
        rest = argv[i + 1:]
        wanted = [a for a in rest if not a.startswith("--")]
        single = set(wanted) if wanted else set()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("加载课程图 ...")
    cg = load_course_graphs()
    print("\n按 TF-IDF 加权 ...")
    scores, df = tfidf_scores(cg)
    print(f"  概念词汇表 {len(df)} 个")
    labels, raw_of = build_name_map(cg)
    print(f"  课程名映射 {len(labels)} 个（重名已加 ①②③ 消歧）\n")
    print("生成可视化 -> output/plots/v01/\n")

    S, ids = plot_course_similarity(cg, OUT_DIR, labels=labels, raw_of=raw_of)
    plot_fingerprints(scores, df, OUT_DIR, labels=labels, raw_of=raw_of)
    plot_heatmap(scores, df, OUT_DIR, labels=labels)
    plot_single_courses(cg, scores, OUT_DIR, only=single, labels=labels, raw_of=raw_of)
    plot_overview(cg, S, ids, df, OUT_DIR, labels=labels)
    cleanup_pyvis_lib()

    # 课程多了之后逐门打印会刷屏，只打前 15 门
    print(f"\n各课程 TF-IDF 最高的 8 个概念（学科指纹，共 {len(scores)} 门，列前 15）:")
    for cid in sorted(scores)[:15]:
        top = " / ".join(c for c, _, _ in scores[cid][:8])
        print(f"  {short_name(cid):22s} {top}")
    if len(scores) > 15:
        print(f"  ...（其余 {len(scores)-15} 门略）")
    print("\n完成。")


if __name__ == "__main__":
    main()
