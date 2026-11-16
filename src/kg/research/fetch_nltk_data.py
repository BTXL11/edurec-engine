"""Fetch the remaining NLTK corpora through the ghproxy GitHub mirror into env/nltk_data."""
import io
import os
import socket
import sys
import time
import urllib.request
import zipfile

socket.setdefaulttimeout(300)
UA = {"User-Agent": "Mozilla/5.0"}
BASE = "https://ghproxy.net/https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env", "nltk_data")

PACKAGES = ["corpora/wordnet.zip", "corpora/omw-1.4.zip"]

for path in PACKAGES:
    dest = os.path.join(ROOT, path.split("/")[0])
    os.makedirs(dest, exist_ok=True)
    target = os.path.join(ROOT, path.split("/")[1].replace(".zip", ""))
    if os.path.isdir(target):
        print(f"SKIP {path} (already present)", flush=True)
        continue
    print(f"GET  {path}", flush=True)
    t0 = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(BASE + path, headers=UA)) as resp:
            blob = resp.read()
        print(f"     downloaded {len(blob)/1048576:.1f} MB in {time.time()-t0:.1f}s", flush=True)
        zipfile.ZipFile(io.BytesIO(blob)).extractall(dest)
        print(f"OK   {path} -> {dest}", flush=True)
    except Exception as exc:
        print(f"FAIL {path}: {type(exc).__name__}: {exc}", flush=True)

# report
for name in ["tokenizers/punkt", "corpora/stopwords", "corpora/wordnet", "corpora/omw-1.4"]:
    p = os.path.join(ROOT, *name.split("/"))
    print(f"{'PRESENT' if os.path.isdir(p) else 'MISSING'} {name}", flush=True)

sys.exit(0)
