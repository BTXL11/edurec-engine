# -*- coding: utf-8 -*-
"""
Local driver: turn the raw Amazon Beauty files into the exact artifacts PRISM's
main.py expects, without touching the repo's own data scripts.

Produces (under data_local/Beauty/):
  reviews_Beauty.txt            user_id  item_id item_id ...   (ordered by time)
  image_features_Beauty.pt      FloatTensor [max_item, 512]
  text_features_Beauty.pt       FloatTensor [max_item, 512]

Item ids follow first-appearance order after the same 5-core filtering that
data/DataProcessing.py performs. Feature tensors are generated with
--feature-source:
  noise   (default) N(0, 0.02) vectors, exactly the placeholder that
          data/process_clip.py substitutes for items without image/text.
  clip    real CLIP ViT-B/32 features computed from downloaded item images
          plus item text (requires the `clip` package and data/Beauty/image_text).
"""
import argparse
import gzip
import json
import os
from collections import defaultdict

import torch


def parse_gz(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as g:
        for line in g:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_interactions(review_gz, out_txt):
    count_u = defaultdict(int)
    count_p = defaultdict(int)

    for rec in parse_gz(review_gz):
        count_u[rec["reviewerID"]] += 1
        count_p[rec["asin"]] += 1

    usermap, itemmap = {}, {}
    users = defaultdict(list)
    for rec in parse_gz(review_gz):
        rev, asin = rec["reviewerID"], rec["asin"]
        if count_u[rev] < 5 or count_p[asin] < 5:
            continue
        uid = usermap.setdefault(rev, len(usermap) + 1)
        iid = itemmap.setdefault(asin, len(itemmap) + 1)
        users[uid].append((iid, float(rec["unixReviewTime"])))

    for uid in users:
        users[uid].sort(key=lambda x: x[1])

    with open(out_txt, "w", encoding="utf-8") as f:
        for uid in sorted(users):
            f.write(str(uid) + " " + " ".join(str(i) for i, _ in users[uid]) + "\n")

    max_item = max(itemmap.values())
    n_inst = sum(len(v) for v in users.values())
    stats = {
        "users": len(users),
        "items": len(itemmap),
        "interactions": n_inst,
        "max_item": max_item,
        "avg_len": n_inst / len(users),
    }
    # asin -> id, needed to line images/text up with ids
    return stats, itemmap, users


def noise_features(max_item, dim, seed=42):
    g = torch.Generator().manual_seed(seed)
    # same distribution process_clip.py falls back to, but [max_item, dim]
    return torch.normal(mean=0.0, std=0.02, size=(max_item, dim), generator=g)


def build_text_features(meta_gz, itemmap, max_item, dim, dataset_dir, device):
    import clip
    from PIL import Image

    text_path = os.path.join(dataset_dir, "image_text", f"Beauty_text.json")
    image_dir = os.path.join(dataset_dir, "image_text", "Beauty_image")
    if not os.path.exists(text_path):
        raise FileNotFoundError(text_path)

    with open(text_path, "r", encoding="utf-8") as fh:
        texts = json.load(fh)

    model, preprocess = clip.load("ViT-B/32", device=device)

    text_feat = noise_features(max_item, dim, seed=1)
    image_feat = noise_features(max_item, dim, seed=2)
    done = 0

    asin_to_id = {a: i for a, i in itemmap.items()}
    for asin, iid in asin_to_id.items():
        img_file = os.path.join(image_dir, f"{asin}.png")
        if asin not in texts or not os.path.exists(img_file):
            continue
        tok = clip.tokenize(texts[asin], truncate=True).to(device)
        img = preprocess(Image.open(img_file).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            tf = model.encode_text(tok)
            imf = model.encode_image(img)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        imf = imf / imf.norm(dim=-1, keepdim=True)
        text_feat[iid - 1] = tf[0].cpu()
        image_feat[iid - 1] = imf[0].cpu()
        done += 1

    return text_feat, image_feat, done


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Beauty")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--feature-source", default="noise", choices=["noise", "clip"])
    args = p.parse_args()

    d = os.path.join(here, args.dataset)
    review_gz = os.path.join(d, f"reviews_{args.dataset}_5.json.gz")
    meta_gz = os.path.join(d, f"meta_{args.dataset}.json.gz")
    out_txt = os.path.join(d, f"reviews_{args.dataset}.txt")

    stats, itemmap, _ = build_interactions(review_gz, out_txt)
    print("interactions:", stats)
    max_item = stats["max_item"]

    with open(os.path.join(d, "itemmap.json"), "w", encoding="utf-8") as fh:
        json.dump(itemmap, fh)
    print("wrote itemmap.json (asin -> item id)")

    if args.feature_source == "clip":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_feat, image_feat, n = build_text_features(
            meta_gz, itemmap, max_item, args.dim, d, device
        )
        print(f"CLIP features built for {n}/{len(itemmap)} items")
    else:
        text_feat = noise_features(max_item, args.dim, seed=1)
        image_feat = noise_features(max_item, args.dim, seed=2)
        print(f"placeholder features: {tuple(text_feat.shape)}")

    torch.save(text_feat, os.path.join(d, f"text_features_{args.dataset}.pt"))
    torch.save(image_feat, os.path.join(d, f"image_features_{args.dataset}.pt"))
    print("wrote", out_txt)
    print("done")


if __name__ == "__main__":
    main()
