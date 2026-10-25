from preprocessing.text import text2RDF
from embeddings.embedder import load_model_, decode_concept
import os
import warnings

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "model")

warnings.filterwarnings("ignore", category=UserWarning)

# 试跑一门课
test_text = "机器学习是人工智能的一个分支，主要研究如何让计算机从数据中学习规律。"
text2RDF(test_text, os.path.join(DATA_ROOT, "output", "graphs", "v01", "TEST_ML"))

# 加载模型查一下
model = load_model_(os.path.join(DATA_ROOT, "models", "wikipedia", "zhwiki_20180420_300d.txt"), 300)

import rdflib

g = rdflib.Graph()
g.parse(os.path.join(DATA_ROOT, "output", "graphs", "v01", "TEST_ML.ttl"), format="turtle")

hits, misses = 0, 0
for s, p, o in g:
    if "pageRank" in str(p):
        # 关键：Wikifier 的 URL 是百分号编码的，必须先解码再查模型
        concept = decode_concept(s)
        try:
            vec = model['ENTITY/' + concept]
            hits += 1
            print(f"✅ {concept}: {vec.shape}")
        except KeyError:
            misses += 1
            print(f"❌ {concept}: 查不到")

total = hits + misses
print(f"\n共 {total} 个概念：命中 {hits}，未命中 {misses}")
if total == 0:
    print("⚠️  图里没有 pageRank 三元组 —— Wikifier 可能没返回结果，检查网络/API 状态")
elif hits == 0:
    print("⚠️  一个都没命中 —— 检查模型路径与实体名解码")
else:
    print("✅ 实体查询链路正常（未命中的多为虚词/单字词，属正常现象）")

