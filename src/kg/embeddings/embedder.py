from gensim.models import KeyedVectors
import numpy as np
import rdflib
import os
from gensim.models import KeyedVectors
from wikipedia2vec import Wikipedia2Vec
from urllib.parse import unquote

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "model")

def symbols_filter(word):
    word = word.replace("%e2%80%93","-")
    word = word.replace("%27", "'")
    word = word.replace("%2f", "/")
    word = word.replace("%e2%80%94", "-")
    word = word.replace("%2e", ".")
    word = word.replace("%26", '&')
    return word


def decode_concept(uri_or_name):
    """把 Wikifier 返回的 URL 片段还原成 Wikipedia2Vec 里的实体名。

    Wikifier 给的 URL 是百分号编码的（如 ``%e6%9c%ba%e5%99%a8%e5%ad%a6%e4%b9%a0``），
    而 ``zhwiki`` 模型里的实体名是干净中文（``机器学习``），所以这里必须先解码、
    再把下划线还原成空格。EMBEDD-ER 的实体查询全部依赖这一步，漏做就会 100% 查不到。
    """
    raw = str(uri_or_name).split("/")[-1]
    raw = unquote(raw)
    raw = raw.replace("_", " ")
    return symbols_filter(raw)

# 原load
def load_model(path, name = 'wikipedia2vec'):
    # dimensions = ["100", "300", "500"]
    print(f"Loading {name} model ...")
    wikipedia2vec = KeyedVectors.load_word2vec_format(path, binary = False)
    print(f"Loaded {name} model ...")
    return wikipedia2vec

# 新load
class Wikipedia2VecWrapper:
    """让 .pkl 模型支持 model[key] 这种字典式查询"""

    def __init__(self, model):
        self.model = model

    def __getitem__(self, key):
        if key.startswith("ENTITY/"):
            name = key[len("ENTITY/"):]
            name = name.replace("_", " ")
            return self.model.get_entity_vector(name)
        else:
            return self.model.get_word_vector(key)

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False


def load_model_(path, name='wikipedia2vec'):
    print(f"Loading {name} model ...")
    pkl_path = path.replace(".txt", ".pkl")
    if os.path.exists(pkl_path):
        model = Wikipedia2Vec.load(pkl_path)
        wrapped = Wikipedia2VecWrapper(model)
        print(f"Loaded {name} model (pkl) ...")
        return wrapped
    # 兜底：没有 pkl 才用 txt
    print(f".pkl 不存在，回退到 .txt（可能很慢）...")
    wikipedia2vec = KeyedVectors.load_word2vec_format(path, binary=False)
    print(f"Loaded {name} model (txt) ...")
    return wikipedia2vec

def wikipedia2vec_embedding(model, concept):
    try :
        return model['ENTITY/'+concept]
    except KeyError:
        return []
        
        
def node_embeddings(model, g, method = 'wikipedia2vec'):
    embeddings_concepts = {}
    concepts = []
    missing_concepts = []
    if method == 'wikipedia2vec':
        # for s, p, o in g:
        #     concept = symbols_filter(s.split('/')[-1])
        for s, p, o in g:
            concept = decode_concept(s)   # URL 解码 + 下划线转空格
            if str(p) == 'https://univ-nantes.fr/ontology/pageRank' :
                embedding = wikipedia2vec_embedding(model, concept)
                if len(embedding) != 0:
                    embeddings_concepts[concept] = {}
                    embeddings_concepts[concept]['list'] = embedding
                    embeddings_concepts[concept]['pageRank'] = float(o)
                    concepts.append(concept)
                else :
                    missing_concepts.append(concept)

    return {
        'embeddings' : embeddings_concepts,
        'concepts' : concepts,
        'missing_concepts' : missing_concepts
    }
    

def embedder_embeddings(resources, d=300):
    path = os.path.join(DATA_ROOT, "models", "wikipedia", "zhwiki_20180420_" + str(d) + "d.txt")
    # 用 load_model_：优先加载 .pkl，避免 .txt 的 EOFError
    model = load_model_(path, d)

    concepts = []
    missing_concepts = []
    sentences_embedder = []

    for c in resources:
        embeddings_concepts = []
        current_concepts = []
        
        g = rdflib.Graph()
        g_path = os.path.join(DATA_ROOT, "output", "graphs", "v01", str(c) + ".ttl")
        try :
            g.parse(g_path, format='turtle')

            embeddings = node_embeddings(model, g, method='wikipedia2vec')

            embeddings_concepts = embeddings['embeddings']
            missing_concepts.append(embeddings['missing_concepts'])
            current_concepts = embeddings['concepts']
            concepts.append(current_concepts)
            pageRankSum = np.sum([embeddings_concepts[k]['pageRank'] for k in embeddings_concepts], 0)
            # 没有命中任何概念时 pageRankSum 为 0，直接相除会得到 NaN
            if pageRankSum == 0:
                print(c, "没有任何可嵌入的概念")
                sentences_embedder.append(np.zeros(d))
                continue
            embeddingsSum = np.sum([np.dot(embeddings_concepts[k]['list'], embeddings_concepts[k]['pageRank']) for k in embeddings_concepts], 0)
            sentences_embedder.append(embeddingsSum / pageRankSum)
        except Exception as e:
            # 原来这里是裸 except，真实错误（例如文件损坏、维度不符）被完全吞掉，
            # 排查起来只能看到 "not found"。保留零向量兜底，但把原因打出来。
            print(c, "not found:", type(e).__name__, e)
            sentences_embedder.append(np.zeros(d))

    return sentences_embedder
