"""从课程/概念/描述生成 RDF 图谱与向量嵌入。

用法示例
--------
# 1. 直接输入一个概念，跑完整流水线（最简单）
python generate.py 实变函数

# 2. 输入课程名 + 描述
python generate.py --title "实变函数与泛函分析基础" \
    --description "程其襄版教材，讲解勒贝格测度与积分。" --author 大表哥

# 3. 多组概念一起处理
python generate.py --concepts 实变函数 泛函分析 拓扑学

# 4. 从文件读（JSON 数组 / CSV / 纯文本每行一条）
python generate.py --input courses.json
python generate.py --input list.txt --method tfidf

# 5. 交互式逐条输入
python generate.py --interactive

产出（默认 output/pipeline/）
    pipeline_result.json   每条资源的图谱统计、概念列表、各方法向量
    resources.csv          平台 resources 表格式，可直接喂 Recommender.from_platform
    concepts.csv           每条资源抽到的概念与 pageRank
    graphs/<key>.ttl       各自的 RDF 图

可选嵌入方法: tfidf / doc2vec / bert / embedder（EMBEDD-ER，需 .ttl 图）
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommender.pipeline import DEFAULT_METHODS, Resource, run  # noqa: E402


def load_input(path):
    """从文件读入资源列表。支持 .json / .csv / 纯文本（每行一条）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return [_resource_from_dict(d) for d in data]
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        with open(path, "r", encoding="utf-8", newline="") as f:
            return [_resource_from_dict(row) for row in csv.DictReader(f, delimiter=sep)]
    # 纯文本：每行一条，行内可用 " | " 分隔 标题|描述|作者
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            title = parts[0]
            desc = parts[1] if len(parts) > 1 else ""
            author = parts[2] if len(parts) > 2 else ""
            out.append(Resource(title=title, description=desc, author=author))
    return out


def _resource_from_dict(d):
    concepts = d.get("concepts") or d.get("concept") or []
    if isinstance(concepts, str):
        concepts = [c.strip() for c in concepts.replace("、", ",").split(",") if c.strip()]
    tags = d.get("tags")
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.replace("、", ",").split(",") if t.strip()]
    meta = d.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    return Resource(
        key=d.get("key") or d.get("id") or None,
        title=d.get("title") or d.get("name") or "",
        concepts=concepts,
        description=d.get("description") or d.get("about") or "",
        author=d.get("author") or "",
        tags=tags,
        metadata=meta,
        source_url=d.get("source_url") or "",
        avg_rating=d.get("avg_rating") or 0.0,
        view_count=d.get("view_count") or 0,
    )


def interactive():
    print("逐条输入，空标题结束。字段可留空。")
    out = []
    while True:
        title = input("\n课程名/概念（空=结束）: ").strip()
        if not title:
            break
        desc = input("描述（可空）: ").strip()
        author = input("作者/UP主（可空）: ").strip()
        out.append(Resource(title=title, description=desc, author=author))
    return out


def build_parser():
    p = argparse.ArgumentParser(
        description="从课程/概念/描述生成 RDF 图谱与向量嵌入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例: python generate.py 实变函数\n"
               "    python generate.py --title 实变函数 --description 讲解勒贝格积分")
    p.add_argument("text", nargs="*",
                   help="直接输入课程名或概念（多个词会拼成一条；多个概念用 --concepts）")
    p.add_argument("--title", default="", help="课程名")
    p.add_argument("--concepts", nargs="+", default=None,
                   help="多个概念/关键词（空格分隔）")
    p.add_argument("--description", "--desc", default="", help="课程描述")
    p.add_argument("--author", default="", help="作者/UP 主")
    p.add_argument("--tags", nargs="+", default=None, help="标签")
    p.add_argument("--input", "-i", help="从文件读（.json / .csv / .txt 每行一条）")
    p.add_argument("--interactive", "-I", action="store_true", help="交互式逐条输入")
    p.add_argument("--method", "-m", nargs="+", default=None,
                   help=f"嵌入方法，默认 {' '.join(DEFAULT_METHODS)}；可选 "
                        "tfidf doc2vec bert embedder")
    p.add_argument("--output", "-o", default=None,
                   help="输出目录（默认 output/pipeline）")
    p.add_argument("--lang", default="zh", help="Wikifier 语言，默认 zh")
    p.add_argument("--threshold", type=float, default=-1,
                   help="Wikifier pageRank 阈值，默认 -1（不筛）")
    p.add_argument("--bert-model", default="bert-base-chinese",
                   help="BERT 模型名，默认 bert-base-chinese")
    p.add_argument("--cpu", action="store_true", help="BERT 强制 CPU")
    p.add_argument("--keep-graph", action="store_true", default=True,
                   help="保留 .ttl 图（默认保留）")
    p.add_argument("--quiet", "-q", action="store_true", help="减少输出")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.interactive:
        resources = interactive()
    elif args.input:
        if not os.path.exists(args.input):
            print(f"找不到输入文件: {args.input}", file=sys.stderr)
            return 2
        resources = load_input(args.input)
    else:
        title = args.title or " ".join(args.text).strip()
        if not title and not args.concepts:
            build_parser().print_help()
            print("\n至少给一个课程名或概念，例如: python generate.py 实变函数",
                  file=sys.stderr)
            return 2
        resources = [Resource(title=title or (args.concepts[0] if args.concepts else ""),
                              concepts=args.concepts,
                              description=args.description,
                              author=args.author,
                              tags=args.tags)]

    # 概念也当成"标题池"：只给了 --concepts 时，标题取第一个概念
    for r in resources:
        if not r.title and r.concepts:
            r.title = r.concepts[0]
        if not r.title:
            print("有一条资源既没有标题也没有概念，已跳过", file=sys.stderr)
    resources = [r for r in resources if r.title]
    if not resources:
        print("没有可处理的资源", file=sys.stderr)
        return 2

    use_cuda = False if args.cpu else None
    report = run(resources,
                 methods=args.method,
                 out_dir=args.output,
                 threshold=args.threshold,
                 lang=args.lang,
                 bert_model=args.bert_model,
                 use_cuda=use_cuda,
                 verbose=not args.quiet)

    # 简明汇总
    print("\n" + "=" * 64)
    print("汇总")
    print("=" * 64)
    for item in report["items"]:
        g = item["graph"]
        names = [c["concept"] for c in g["concepts"][:6]]
        print(f"[{item['key']}] {item['title']}")
        print(f"    图谱: {g['entities']} 实体 / {g['triples']} 三元组")
        if names:
            print(f"    概念: {' / '.join(names)}")
        for m, e in item["embeddings"].items():
            print(f"    {m:9s} 维度={e['dim']}")
    print(f"\n结果目录: {args.output or os.path.join('output', 'pipeline')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
