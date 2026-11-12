# -*- coding: utf-8 -*-
"""
Collect every run under outputs/ into one markdown table.

Parses the per-run log written by main.py/trainers.py:
  {'Epoch': 'best', ...}       validation  (first  occurrence)
  {'Epoch': 'best', ...}       test        (second occurrence)
"""
import os
import re
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABELS = {
    "Beauty_idonly":          "① 纯 ID STOSA（256/100）",
    "Beauty_clip_noPRISM":    "② +图文直加（256/100）",
    "Beauty_placeholder":     "③ PRISM + 噪声特征（256/100）",
    "Beauty_clip":            "④ PRISM + 真实 CLIP（256/100）",
    "small_idonly":           "纯 ID STOSA（64/50）",
    "small_prism":            "PRISM（64/50, λ=0.1）",
    "small_lam_tuned":        "PRISM（64/50, λ=0.05/0.2/0.5）",
    "small_fusiononly":       "PRISM（64/50, fusion_only）",
    "small_fusiononfy_lam":   "PRISM（64/50, fusion_only+调λ）",
    "small_prism_long":       "PRISM（64/50, 长训 150ep）",
    "seed1_idonly":           "纯 ID STOSA（64/50, seed 1）",
    "seed2_idonly":           "纯 ID STOSA（64/50, seed 2）",
    "seed1_prism":            "PRISM（64/50, seed 1）",
    "seed2_prism":            "PRISM（64/50, seed 2）",
}

FIELDS = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"]


def parse_log(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    best = re.findall(r"\{'Epoch': 'best'.*?\}", txt)
    vals = []
    for block in best[:2]:
        row = {}
        for f in FIELDS:
            m = re.search(rf"'{re.escape(f)}': '([-0-9.]+)'", block)
            row[f] = float(m.group(1)) if m else None
        vals.append(row)
    total = re.search(r"Total training time: ([\d:]+)", txt)
    return (vals + [None, None])[:2], (total.group(1) if total else "-")


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "outputs", "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        logs = glob.glob(os.path.join(d, "*.txt"))
        if not logs:
            continue
        (val, test), total = parse_log(logs[0])
        rows.append((LABELS.get(name, name), val, test, total))

    hdr = "| 变体 | 验证 R@10 | 验证 N@10 | 测试 R@10 | 测试 R@20 | 测试 N@10 | 测试 N@20 | 时长 |"
    sep = "|---|---|---|---|---|---|---|---|"
    print(hdr)
    print(sep)
    for label, val, test, total in rows:
        if not test:
            continue
        print(f"| {label} | {val['Recall@10']:.4f} | {val['NDCG@10']:.4f} | "
              f"{test['Recall@10']:.4f} | {test['Recall@20']:.4f} | "
              f"{test['NDCG@10']:.4f} | {test['NDCG@20']:.4f} | {total} |")


if __name__ == "__main__":
    main()
