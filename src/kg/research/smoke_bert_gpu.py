"""Smoke-test the project's deep-learning embedding path on the GPU.

Reproduces what embeddings/bert.py + embeddings/methods.py do:
    RepresentationModel("bert", "prajjwal1/bert-tiny").encode_sentences([...])
using the hf-mirror.com endpoint, so it also proves the model download and
torch-on-GPU path work end to end.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, "env", "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(HERE, "env", "torch"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, "env", "matplotlib"))
os.environ.setdefault("NLTK_DATA", os.path.join(HERE, "env", "nltk_data"))

for key in ("HF_HOME", "TORCH_HOME", "MPLCONFIGDIR", "NLTK_DATA"):
    os.makedirs(os.environ[key], exist_ok=True)

import torch  # noqa: E402
from simpletransformers.language_representation import RepresentationModel  # noqa: E402

sentences = [
    "Linear algebra is the branch of mathematics concerning linear equations.",
    "This lecture introduces machine learning and supervised classification.",
    "Databases store structured data and support SQL queries.",
]

print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}, "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")

t0 = time.time()
model = RepresentationModel(
    "bert",
    "prajjwal1/bert-tiny",
    use_cuda=torch.cuda.is_available(),
    args={"no_save": True, "silent": True},
)
print(f"model loaded in {time.time()-t0:.1f}s")

t0 = time.time()
embeddings = model.encode_sentences(sentences, combine_strategy="mean")
print(f"encoded {len(sentences)} sentences in {time.time()-t0:.2f}s")

shape = getattr(embeddings, "shape", None)
print("embedding type:", type(embeddings).__name__, "shape:", shape)
if shape is None:
    print("FAIL: unexpected embedding type")
    sys.exit(1)

assert shape[0] == len(sentences), f"expected {len(sentences)} rows, got {shape}"
assert shape[1] == 128, f"bert-tiny should give 128 dims, got {shape[1]}"
print("first row (first 5 dims):", [round(float(v), 4) for v in embeddings[0][:5]])

# the GPU really did the work
if torch.cuda.is_available():
    print(f"GPU memory in use: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    assert torch.cuda.memory_allocated() > 0, "no GPU memory used - ran on CPU"
    print("confirmed: embeddings were computed on the GPU")

print("EMBEDDING PIPELINE OK")
