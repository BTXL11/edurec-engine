"""批量跑 MOOCCube 课程 -> RDF 图。

与 Generation.ipynb 的 DataFrame 构造完全一致（course.json -> 按 core_id 聚合 ->
拼 Text -> 清洗文件名），额外提供：

* 固定的"前 N 条"定义：按聚合后的顺序取前 N，并把 Cid 清单落盘到
  output/graphs/v01/_codes.json，保证多次运行处理的是同一批课程；
* 断点续传：已存在且非空的 .ttl 直接跳过（沿用 safe_text2RDF 的现有逻辑）；
* 逐条计时与结果汇总。

用法:
    python run_batch.py            # 前 20 条
    python run_batch.py 50         # 前 50 条
    python run_batch.py --all      # 全部 583 条
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

from preprocessing.text import safe_text2RDF  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
VERSION = "v01"
GRAPH_DIR = os.path.join(ROOT, "output", "graphs", VERSION)
CODES_FILE = os.path.join(GRAPH_DIR, "_codes.json")


def build_df_chapters():
    """与 notebook cell 2 相同的构造流程，但用绝对路径读数据。"""
    path = os.path.join(ROOT, "data", "MOOCCube", "entities", "course.json")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    df_raw = pd.DataFrame(records)

    def flatten_cell(v):
        if isinstance(v, list):
            return " ".join(flatten_cell(i) for i in v if i is not None)
        return str(v) if v is not None else ""

    for col in ["name", "about", "chapter", "display_name"]:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].apply(flatten_cell)

    df_chapters = (
        df_raw.groupby("core_id", as_index=False)
        .agg({
            "name": "first",
            "about": "first",
            "chapter": lambda x: " ".join(x.dropna().unique()),
            "display_name": lambda x: " ".join(x.dropna().unique()),
        })
        .rename(columns={"core_id": "Cid"})
    )

    def safe_filename(s):
        return re.sub(r'[\\/:*?"<>|]', "_", str(s))

    df_chapters["Cid_raw"] = df_chapters["Cid"]
    df_chapters["Cid"] = df_chapters["Cid"].apply(safe_filename)
    df_chapters["Text"] = (
        df_chapters["name"].fillna("") + ". " +
        df_chapters["about"].fillna("") + ". " +
        df_chapters["chapter"].fillna("") + ". " +
        df_chapters["display_name"].fillna("")
    ).str.strip()
    return df_chapters[["Cid", "Cid_raw", "Text"]]


def select_batch(df, limit, codes_file):
    """取前 limit 条。若清单已存在，则严格按清单跑（保证可复现）。"""
    if os.path.exists(codes_file):
        with open(codes_file, "r", encoding="utf-8") as f:
            codes = json.load(f)
        df = df.set_index("Cid", drop=False)
        missing = [c for c in codes if c not in df.index]
        if missing:
            print(f"警告：清单里有 {len(missing)} 条在数据中找不到，忽略")
        df = df.loc[[c for c in codes if c in df.index]].reset_index(drop=True)
        print(f"使用已存在的清单 {os.path.basename(codes_file)}（{len(df)} 条）")
        return df
    return df.head(limit)


def main():
    args = [a for a in sys.argv[1:]]
    run_all = "--all" in args
    limit = int(args[0]) if (args and args[0].isdigit()) else 20

    os.makedirs(GRAPH_DIR, exist_ok=True)
    df = build_df_chapters()
    if run_all:
        limit = len(df)
    print(f"聚合后课程总数: {len(df)}，本次处理前 {limit} 条\n")

    batch = select_batch(df, limit, CODES_FILE)
    if not os.path.exists(CODES_FILE):
        with open(CODES_FILE, "w", encoding="utf-8") as f:
            json.dump(list(batch["Cid"]), f, ensure_ascii=False, indent=1)
        print(f"已写入清单 {CODES_FILE}（{len(batch)} 条）\n")

    ok, skipped, failed = [], [], []
    t_start = time.time()
    for i, (_, row) in enumerate(batch.iterrows(), 1):
        cid = row["Cid"]
        ttl = os.path.join(GRAPH_DIR, cid + ".ttl")
        if os.path.exists(ttl) and os.path.getsize(ttl) > 0:
            skipped.append(cid)
            print(f"[{i}/{len(batch)}] 跳过（已存在）: {cid}")
            continue

        print(f"[{i}/{len(batch)}] 处理中: {cid}  (文本 {len(row['Text'])} 字)")
        t0 = time.time()
        try:
            g = safe_text2RDF(row, version=VERSION)
            elapsed = time.time() - t0
            if g is not None and os.path.exists(ttl) and os.path.getsize(ttl) > 0:
                n = len(g)
                ok.append((cid, n, elapsed))
                print(f"          ✅ 成功: {n} 个三元组, 耗时 {elapsed:.1f}s")
            else:
                failed.append((cid, "Wikifier 未返回可用实体"))
                print(f"          ❌ 失败: Wikifier 未返回可用实体, 耗时 {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.time() - t0
            failed.append((cid, f"{type(exc).__name__}: {exc}"))
            print(f"          ❌ 异常: {type(exc).__name__}: {exc}, 耗时 {elapsed:.1f}s")

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 66)
    print(f"完成: 成功 {len(ok)}，跳过 {len(skipped)}，失败 {len(failed)}，"
          f"总计 {len(batch)} 条，用时 {total_elapsed/60:.1f} 分钟")
    if ok:
        avg = sum(t for _, _, t in ok) / len(ok)
        print(f"平均每条 {avg:.1f}s；估算剩余 {len(df)-len(ok)-len(skipped)} 条 "
              f"约需 {avg*(len(df)-len(ok)-len(skipped))/60:.0f} 分钟")
        print("三元组数最少/最多: %d / %d" % (min(n for _, n, _ in ok), max(n for _, n, _ in ok)))
    if skipped:
        print(f"跳过 {len(skipped)} 条: {', '.join(skipped[:5])}{' ...' if len(skipped) > 5 else ''}")
    if failed:
        print("失败清单:")
        for cid, why in failed:
            print(f"  - {cid}: {why}")
    print("=" * 66)


if __name__ == "__main__":
    main()
