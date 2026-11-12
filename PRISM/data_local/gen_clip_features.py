# -*- coding: utf-8 -*-
"""
Generate the real 512-d CLIP ViT-B/32 item features that PRISM's main.py expects,
mirroring data/Image_download.py + data/process_clip.py.

Inputs (all under data_local/Beauty/):
  itemmap.json                 asin -> PRISM item id (from prepare_data.py)
  metadata_5core.parquet       asin + merged item text for the 5-core items
  metadata_full.parquet        full metadata (title/description/brand/categories/imUrl)
  images/<asin>.jpg            download cache (filled in by this script)

Outputs:
  text_features_Beauty.pt      FloatTensor [max_item, 512], L2-normalised
  image_features_Beauty.pt     FloatTensor [max_item, 512], L2-normalised

Items without text or whose image could not be fetched keep the N(0, 0.02)
placeholder that process_clip.py itself substitutes, so the code path is identical.

CLIP itself is used straight from the vendored openai/CLIP checkout (no pip):
  _vendor/clip_repo  +  _vendor/site (ftfy)  +  _vendor/ViT-B-32.pt
"""
import argparse
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch
import urllib.request
import ssl

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data_local", "Beauty")
CTX = ssl._create_unverified_context()
UA = {"User-Agent": "Mozilla/5.0"}


def setup_clip():
    sys.path.insert(0, os.path.join(HERE, "_vendor", "site"))
    sys.path.insert(0, os.path.join(HERE, "_vendor", "clip_repo"))
    import clip
    return clip


def image_url(u):
    """2014 imUrl -> still-live Amazon media CDN form."""
    m = re.search(r"/images/I/([^/]+)$", u or "")
    return f"https://m.media-amazon.com/images/I/{m.group(1)}" if m else None


def fetch_one(asin, url, out_dir):
    path = os.path.join(out_dir, f"{asin}.jpg")
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return asin, True
    for attempt in (0, 1):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=12, context=CTX)
            data = r.read()
            if len(data) > 1024:
                with open(path, "wb") as fh:
                    fh.write(data)
                return asin, True
        except Exception:
            if attempt == 0:
                time.sleep(0.3)
    return asin, False


def download_images(asin_to_url, out_dir, workers, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    todo = list(asin_to_url.items())
    if limit:
        todo = todo[:limit]
    todo = [(a, u) for a, u in todo if not (os.path.exists(os.path.join(out_dir, f"{a}.jpg")))]
    print(f"downloading {len(todo)} images with {workers} workers", flush=True)
    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_one, a, u, out_dir) for a, u in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            _, good = fut.result()
            ok += good
            fail += (not good)
            if i % 1000 == 0:
                print(f"  {i}/{len(todo)}  ok={ok} fail={fail}  {i/max(time.time()-t0,1e-6):.1f} img/s", flush=True)
    print(f"images done: ok={ok} fail={fail} in {time.time()-t0:.0f}s", flush=True)
    return ok, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--image-limit", type=int, default=0, help="0 = all")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    clip = setup_clip()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(DATA, "itemmap.json"), encoding="utf-8") as fh:
        asin_to_id = json.load(fh)
    max_item = max(asin_to_id.values())
    dim = 512

    # ---- text
    core = pd.read_parquet(os.path.join(DATA, "metadata_5core.parquet"))
    core_text = {str(a): str(t) for a, t in zip(core["asin"], core["text"])}

    full = pd.read_parquet(
        os.path.join(DATA, "metadata_full.parquet"),
        columns=["asin", "title", "description", "brand", "categories", "imUrl"],
    )
    full = full[full["asin"].astype(str).isin(asin_to_id)]
    full_by_asin = {str(r.asin): r for r in full.itertuples()}

    def build_text(asin):
        t = core_text.get(asin)
        if t and len(t) > 0:
            return t
        r = full_by_asin.get(asin)
        if r is None:
            return None
        cats = r.categories
        if isinstance(cats, (list, tuple)) and len(cats):
            cats = cats[0] if isinstance(cats[0], (list, tuple)) else cats
            cats = " > ".join(map(str, cats))
        parts = [str(x) for x in (r.title, r.description, r.brand, cats) if isinstance(x, str) and x]
        return " ".join(parts) if parts else None

    texts, missing_text = {}, []
    for asin in asin_to_id:
        t = build_text(asin)
        if t:
            texts[asin] = t
        else:
            missing_text.append(asin)
    print(f"text available for {len(texts)}/{len(asin_to_id)} items ({len(missing_text)} missing)", flush=True)

    # ---- images
    asin_to_url = {}
    for asin, r in full_by_asin.items():
        u = image_url(r.imUrl)
        if u:
            asin_to_url[asin] = u
    if not args.skip_download:
        download_images(asin_to_url, os.path.join(DATA, "images"), args.workers,
                        limit=args.image_limit or None)
    have_img = {a for a in asin_to_url if os.path.exists(os.path.join(DATA, "images", f"{a}.jpg"))}
    print(f"images on disk: {len(have_img)}", flush=True)

    # ---- encode
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=os.path.join(HERE, "_vendor"))
    model.eval()

    g = torch.Generator().manual_seed(42)
    text_feat = torch.normal(0.0, 0.02, size=(max_item, dim), generator=g)
    image_feat = torch.normal(0.0, 0.02, size=(max_item, dim), generator=g)

    asins = list(texts)
    print(f"encoding {len(asins)} texts ...", flush=True)
    with torch.no_grad():
        for i in range(0, len(asins), args.batch):
            chunk = asins[i:i + args.batch]
            tok = clip.tokenize([texts[a] for a in chunk], truncate=True).to(device)
            f = model.encode_text(tok).float()
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            for a, vec in zip(chunk, f.cpu()):
                text_feat[asin_to_id[a] - 1] = vec
            if (i // args.batch) % 20 == 0:
                print(f"  text {i + len(chunk)}/{len(asins)}", flush=True)

    from PIL import Image
    img_asins = [a for a in asins if a in have_img]
    print(f"encoding {len(img_asins)} images ...", flush=True)
    done_img = 0
    with torch.no_grad():
        for i in range(0, len(img_asins), args.batch):
            chunk = img_asins[i:i + args.batch]
            tensors, keep = [], []
            for a in chunk:
                try:
                    im = Image.open(os.path.join(DATA, "images", f"{a}.jpg")).convert("RGB")
                    tensors.append(preprocess(im))
                    keep.append(a)
                except Exception:
                    continue
            if not tensors:
                continue
            batch = torch.stack(tensors).to(device)
            f = model.encode_image(batch).float()
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            for a, vec in zip(keep, f.cpu()):
                image_feat[asin_to_id[a] - 1] = vec
            done_img += len(keep)
            if (i // args.batch) % 20 == 0:
                print(f"  image {i + len(chunk)}/{len(img_asins)}", flush=True)

    cov_t = len(texts) / len(asin_to_id) * 100
    cov_i = done_img / len(asin_to_id) * 100
    print(f"coverage: text {cov_t:.1f}%  image {cov_i:.1f}%", flush=True)

    torch.save(text_feat, os.path.join(DATA, "text_features_Beauty.pt"))
    torch.save(image_feat, os.path.join(DATA, "image_features_Beauty.pt"))
    print("saved real CLIP features")


if __name__ == "__main__":
    main()
