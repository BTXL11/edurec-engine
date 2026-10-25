"""GPT-2 句向量。

HF_ENDPOINT 必须在 import transformers 之前设置：本机 huggingface.co 被
加速器用自签证书中间人代理，Python 直连必然报 CERTIFICATE_VERIFY_FAILED。
详见 embeddings/bert.py 的说明。
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "env", "huggingface")
)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from simpletransformers.language_representation import RepresentationModel  # noqa: E402


def gpt2_embeddings(sentences, model_name="gpt2-large", use_cuda=None):
    if use_cuda is None:
        import torch

        use_cuda = torch.cuda.is_available()
    model_gpt2 = RepresentationModel(
        model_type = "gpt2",
        model_name = model_name,
        use_cuda = use_cuda
    )
    sentences_vector_gpt2 = list(model_gpt2.encode_sentences(sentences, combine_strategy = "mean"))
    
    return sentences_vector_gpt2
