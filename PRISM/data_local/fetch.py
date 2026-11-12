import os
import ssl
import sys
import time
import urllib.request

ctx = ssl._create_unverified_context()
UA = {"User-Agent": "Mozilla/5.0"}


def size_of(url):
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    try:
        r = urllib.request.urlopen(req, timeout=30, context=ctx)
        return int(r.headers.get("Content-Length") or 0)
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}"


def download(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
        print(f"skip {out} ({os.path.getsize(out)/1048576:.1f} MB)", flush=True)
        return True
    t0 = time.time()
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r, open(out, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 10:
                last = time.time()
                pct = 100 * done / total if total else 0
                print(f"  {done/1048576:.1f}/{total/1048576:.1f} MB ({pct:.0f}%) "
                      f"{done/1024/max(time.time()-t0,1e-6):.0f} KB/s", flush=True)
    print(f"done {out} {os.path.getsize(out)/1048576:.1f} MB in {time.time()-t0:.0f}s", flush=True)
    return True


if __name__ == "__main__":
    base = "https://hf-mirror.com/datasets"
    here = os.path.dirname(os.path.abspath(__file__))
    beauty = os.path.join(here, "Beauty")
    os.makedirs(beauty, exist_ok=True)

    download(f"{base}/ttb06/amazon_beauty_5_2014/resolve/main/reviews_Beauty_5.json.gz",
             os.path.join(beauty, "reviews_Beauty_5.json.gz"))

    # report sizes of the optional CLIP feature files before pulling them
    for name in ["item2idx.pt", "idx_meta_map.pt", "idx_text_emb_map_clip.pt", "idx_img_emb_map_clip.pt"]:
        u = f"{base}/SLKpnu/amazon_2014_5_core/resolve/main/{name}"
        s = size_of(u)
        if isinstance(s, int):
            print(f"{name:32s} {s/1048576:8.1f} MB", flush=True)
        else:
            print(f"{name:32s} {s}", flush=True)
