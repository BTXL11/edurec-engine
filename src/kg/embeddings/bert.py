"""BERT 句向量。

关于 HF_ENDPOINT：本机无法直连 huggingface.co（该域名会被 Steam++ 之类的
加速器用自签根证书中间人代理，Python 校验必然失败，报
`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`）。
所以这里在 import transformers 之前就把端点兜底到 hf-mirror.com，
不依赖外部环境变量是否设置 —— 否则在没走 setup_env.ps1 的 Jupyter 内核里必然复现该报错。
"""
import os

# 必须在 import transformers / simpletransformers 之前设置：transformers 在导入时读取端点
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 缓存放到项目内，避免往 %USERPROFILE% 写（本机该目录不可写）
os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "env", "huggingface")
)
# Windows 未开开发者模式时无法建符号链接，huggingface_hub 会因此打一大段警告
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from simpletransformers.language_representation import RepresentationModel  # noqa: E402


def bert_embeddings(sentences, model_name="bert-base-uncased", use_cuda=None):
    if use_cuda is None:
        import torch

        use_cuda = torch.cuda.is_available()
    model_bert = RepresentationModel(
        model_type = "bert",
        model_name = model_name,
        use_cuda = use_cuda
    )
    sentences_vector_bert = list(model_bert.encode_sentences(sentences, combine_strategy = "mean"))
    
    return sentences_vector_bert
