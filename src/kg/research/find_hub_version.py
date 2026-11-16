"""Find a huggingface_hub version compatible with transformers 4.25.1 on Windows.

huggingface_hub 0.11.1 uses the (quoted) ETag verbatim in lock/cache file names,
which Windows rejects (WinError 123). Newer 0.1x releases sanitise it, but
transformers 4.25.1 requires huggingface-hub<1.0,>=0.10.0, so several 0.1x
versions are allowed. This probes them and keeps the first that works.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"

CANDIDATES = ["0.11.1", "0.13.4", "0.15.1", "0.16.4", "0.17.3", "0.19.4"]

PROBE = r"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "env", "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "env", "matplotlib"))
os.environ.setdefault("NLTK_DATA", os.path.join(os.getcwd(), "env", "nltk_data"))
import huggingface_hub, transformers, torch
print("hub", huggingface_hub.__version__, "transformers", transformers.__version__)
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
mdl = AutoModel.from_pretrained("prajjwal1/bert-tiny")
import torch
enc = tok(["hello world"], return_tensors="pt")
with torch.no_grad():
    out = mdl(**enc)
print("hidden", tuple(out.last_hidden_state.shape))
print("PROBE_OK")
"""


def run(cmd):
    return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=900)


for ver in CANDIDATES:
    print(f"=== huggingface_hub=={ver} ===", flush=True)
    inst = run([PY, "-m", "pip", "install", "--no-input", "--progress-bar", "off",
                "--no-cache-dir", "--index-url", INDEX, f"huggingface-hub=={ver}"])
    if inst.returncode != 0:
        print("  install failed:", inst.stdout[-400:], inst.stderr[-400:], flush=True)
        continue
    probe = run([PY, "-c", PROBE])
    tail = "\n".join(probe.stdout.strip().splitlines()[-4:])
    if "PROBE_OK" in probe.stdout:
        print("  WORKS ->", tail, flush=True)
        print(f"KEEP huggingface_hub=={ver}")
        sys.exit(0)
    err = [ln for ln in (probe.stdout + probe.stderr).splitlines() if "Error" in ln or "error" in ln]
    print("  failed ->", (err[-1] if err else tail)[:300], flush=True)

print("no candidate worked")
sys.exit(1)
