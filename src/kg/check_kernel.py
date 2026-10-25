"""End-to-end check: run a notebook against the registered EMBEDD-ER kernel.

Proves the kernelspec launches and that the project's own modules import inside
a real Jupyter kernel process (not just in a plain python -c run).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["JUPYTER_DATA_DIR"] = os.path.join(HERE, "env", "jupyter")
os.environ["JUPYTER_CONFIG_DIR"] = os.path.join(HERE, "env", "jupyter", "config")
os.environ["JUPYTER_RUNTIME_DIR"] = os.path.join(HERE, "env", "jupyter", "runtime")
os.environ["IPYTHONDIR"] = os.path.join(HERE, "env", "ipython")
os.environ["MPLCONFIGDIR"] = os.path.join(HERE, "env", "matplotlib")
os.environ["NLTK_DATA"] = os.path.join(HERE, "env", "nltk_data")
os.makedirs(os.environ["JUPYTER_RUNTIME_DIR"], exist_ok=True)

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

SOURCE = r"""
import sys, os
print("kernel python:", sys.executable)
import numpy as np, pandas as pd, rdflib, gensim, sklearn, matplotlib, networkx, pyvis, nltk
import torch
print("numpy", np.__version__, "| pandas", pd.__version__, "| sklearn", sklearn.__version__)
print("rdflib", rdflib.__version__, "| gensim", gensim.__version__)
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
from embeddings.methods import embeddings
from preprocessing.text import text2RDF
from similarity.metrics import distance_matrix
from classification.classification import classify_cv
from plotting.graph import visualise_rdf_graph
print("project modules imported OK")
from similarity.metrics import distance_matrix
X = np.random.rand(6, 8)
print("cosine matrix shape:", np.asarray(distance_matrix(list(X), method="cosine")).shape)
from embeddings.tfidf import tfidf_embeddings
print("tfidf ok")
"""

nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_code_cell(SOURCE)]
nb.metadata["kernelspec"] = {
    "display_name": "Python 3.10 (EMBEDD-ER)",
    "language": "python",
    "name": "embedd-er",
}

client = NotebookClient(nb, timeout=600, kernel_name="embedd-er",
                        resources={"metadata": {"path": HERE}})
client.execute()

failed = False
for cell in nb.cells:
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            sys.stdout.write(out.get("text", ""))
        elif out.get("output_type") == "error":
            failed = True
            print("ERROR:", out.get("ename"), out.get("evalue"))
            for line in out.get("traceback", [])[-6:]:
                print(line)
print("NOTEBOOK KERNEL CHECK:", "FAILED" if failed else "OK")
sys.exit(1 if failed else 0)
