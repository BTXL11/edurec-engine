import ast
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

from preprocessing.text import safe_text2RDF  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
VERSION = "v02"
GRAPH_DIR = os.path.join(ROOT, "output", "graphs", VERSION)
CODES_FILE = os.path.join(GRAPH_DIR, "_codes.json")
DATA_PATH = os.path.join(ROOT, "data", "数学建模.json")


# ---------------------------------------------------------------- 数据读取 ---
def _parse_line(line: str, lineno: int):
    """解析一行：优先标准 JSON，退化为 Python 字面量。"""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(line)
    except (ValueError, SyntaxError):
        # 最后兜底：只放行 nan / inf / None / True / False，不暴露 __builtins__
        try:
            return eval(
                line,
                {"__builtins__": {}},
                {
                    "nan": float("nan"),
                    "inf": float("inf"),
                    "None": None,
                    "True": True,
                    "False": False,
                },
            )
        except Exception as exc:
            raise ValueError(
                f"第 {lineno} 行无法解析为 JSON / Python 字面量: {line[:200]!r}"
            ) from exc


def _extract_rows(data):
    """从 {table, rows} / 纯 list / 任意含 list 的 dict 中抽出记录列表。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # 优先 rows
        if isinstance(data.get("rows"), list):
            return data["rows"]
        # 常见包装 key
        for key in ("data", "records", "items", "list", "result",
                    "results", "chapters"):
            if isinstance(data.get(key), list):
                return data[key]
        # 兜底：取任意一个 list 字段
        for v in data.values():
            if isinstance(v, list):
                return v
        return [data]
    return [data]


def _load_rows(path: str) -> list[dict]:
    """读取数据文件，兼容：合法 JSON、Python 字面量、JSONL 三种格式。"""
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    stripped = raw.strip()
    if not stripped:
        return []

    # 1) 整体是合法 JSON
    try:
        return _extract_rows(json.loads(stripped))
    except json.JSONDecodeError:
        pass

    # 2) 整体是 Python 字面量（如 str(records) 写出的文件）
    try:
        return _extract_rows(ast.literal_eval(stripped))
    except (ValueError, SyntaxError):
        pass

    # 3) 逐行解析（JSONL / 每行一个 dict repr）
    records = []
    for i, line in enumerate(stripped.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        records.append(_parse_line(line, i))

    # 若某行解析出来仍是字符串，再解一层
    normalized = []
    for r in records:
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except json.JSONDecodeError:
                pass
        normalized.append(r)
    return _extract_rows(normalized)


def _maybe_json(v):
    """把形如 '[..]' / '{..}' 的字符串尝试解析成 Python 对象。"""
    if isinstance(v, str):
        s = v.strip()
        if s and s[0] in "[{":
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
    return v


def _flatten_cell(v):
    """把 list / dict / 标量统一展平成字符串。"""
    if isinstance(v, list):
        return " ".join(_flatten_cell(i) for i in v if i is not None)
    if isinstance(v, dict):
        return " ".join(f"{k}:{v[k]}" for k in v)
    return str(v) if v is not None else ""


# ---------------------------------------------------------------- DataFrame ---
def build_df_chapters():
    records = _load_rows(DATA_PATH)
    df_raw = pd.DataFrame(records)

    # tags / metadata 里是字符串形式的 JSON，先解析回对象
    for col in ["tags", "metadata"]:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].apply(_maybe_json)

    # 展平参与 Text 拼接的字段
    for col in ["title", "description", "type", "tags", "author"]:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].apply(_flatten_cell)

    # 按 id 聚合
    agg_map = {
        col: how
        for col, how in [
            ("title", "first"),
            ("author", "first"),
            ("description", "first"),
            ("tags", lambda x: " ".join(x.dropna().unique())),
            ("type", lambda x: " ".join(x.dropna().unique())),
        ]
        if col in df_raw.columns
    }

    df_chapters = (
        df_raw.groupby("id", as_index=False)
        .agg(agg_map)
        .rename(columns={"id": "Cid"})
    )
    for col in ["title", "description", "tags", "type"]:
        if col not in df_chapters.columns:
            df_chapters[col] = ""

    def safe_filename(s):
        s = str(s) if s is not None else ""
        if not s:
            return "unknown"
        return re.sub(r'[\\/:*?"<>|]', "_", s)

    df_chapters["Cid_raw"] = df_chapters["Cid"]
    df_chapters["Cid"] = df_chapters["Cid"].apply(safe_filename)
    df_chapters["Text"] = (
        df_chapters["title"].fillna("").astype(str) + ". "
        + df_chapters["description"].fillna("").astype(str) + ". "
        + df_chapters["tags"].fillna("").astype(str) + ". "
        + df_chapters["type"].fillna("").astype(str)
    ).str.strip()

    return df_chapters[["Cid", "Cid_raw", "Text"]]


# ---------------------------------------------------------------- 批次选择 ---
def select_batch(df, limit, codes_file):
    """取前 limit 条。若清单已存在，则严格按清单跑（保证可复现）。

    清单里存的是 `Cid_raw`（原始 id），避免不同 id 经过 safe_filename
    清洗后撞名导致的丢失。
    """
    if os.path.exists(codes_file):
        with open(codes_file, "r", encoding="utf-8") as f:
            codes = [str(c) for c in json.load(f)]

        df = df.copy()
        df["_key"] = df["Cid_raw"].astype(str)
        idx = df.set_index("_key", drop=False)

        missing = [c for c in codes if c not in idx.index]
        if missing:
            print(f"警告：清单里有 {len(missing)} 条在数据中找不到，忽略")

        out = idx.loc[[c for c in codes if c in idx.index]]
        out = out.drop(columns=["_key"], errors="ignore").reset_index(drop=True)
        print(f"使用已存在的清单 {os.path.basename(codes_file)}（{len(out)} 条）")
        return out

    return df.head(limit)


# ---------------------------------------------------------------- 主流程 ---
def main():
    args = sys.argv[1:]
    run_all = "--all" in args
    # 支持 "50 --all" 这种写法
    limit_arg = next((int(a) for a in args if a.isdigit()), 20)
    limit = limit_arg


    os.makedirs(GRAPH_DIR, exist_ok=True)
    df = build_df_chapters()
    if run_all:
        # --all：忽略旧清单，全量跑，并把清单重写为全量
        batch = df.reset_index(drop=True)
        with open(CODES_FILE, "w", encoding="utf-8") as f:
            json.dump([str(x) for x in batch["Cid_raw"]],
                      f, ensure_ascii=False, indent=1)
        print(f"已覆盖写入清单 {CODES_FILE}（{len(batch)} 条）\n")
    else:
        batch = select_batch(df, limit, CODES_FILE)
        if not os.path.exists(CODES_FILE):
            with open(CODES_FILE, "w", encoding="utf-8") as f:
                json.dump([str(x) for x in batch["Cid_raw"]],
                          f, ensure_ascii=False, indent=1)
            print(f"已写入清单 {CODES_FILE}（{len(batch)} 条）\n")

    batch = select_batch(df, limit, CODES_FILE)
    if not os.path.exists(CODES_FILE):
        with open(CODES_FILE, "w", encoding="utf-8") as f:
            json.dump(
                [str(x) for x in batch["Cid_raw"]],
                f,
                ensure_ascii=False,
                indent=1,
            )
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
    print(
        f"完成: 成功 {len(ok)}，跳过 {len(skipped)}，失败 {len(failed)}，"
        f"总计 {len(batch)} 条，用时 {total_elapsed / 60:.1f} 分钟"
    )
    if ok:
        avg = sum(t for _, _, t in ok) / len(ok)
        remaining = len(df) - len(ok) - len(skipped)
        print(
            f"平均每条 {avg:.1f}s；估算剩余 {remaining} 条 "
            f"约需 {avg * max(remaining, 0) / 60:.0f} 分钟"
        )
        print(
            "三元组数最少/最多: %d / %d"
            % (min(n for _, n, _ in ok), max(n for _, n, _ in ok))
        )
    if skipped:
        print(
            f"跳过 {len(skipped)} 条: {', '.join(skipped[:5])}"
            f"{' ...' if len(skipped) > 5 else ''}"
        )
    if failed:
        print("失败清单:")
        for cid, why in failed:
            print(f"  - {cid}: {why}")
    print("=" * 66)


if __name__ == "__main__":
    main()