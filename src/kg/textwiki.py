from wikipedia2vec import Wikipedia2Vec
import os
import warnings

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "model")
warnings.filterwarnings("ignore", category=UserWarning)
model = Wikipedia2Vec.load(os.path.join(DATA_ROOT, "models", "wikipedia", "zhwiki_20180420_300d.pkl"))

# 直接测试常见实体能不能查到
for name in ["人工智能", "机器学习", "计算机科学", "数学", "数据", "深度学习"]:
    try:
        vec = model.get_entity_vector(name)
        print(f"✅ {name}: {vec.shape}")
    except KeyError:
        print(f"❌ {name}: 查不到")

# 也可以看看模型里到底存了哪些实体名（取前 20 个）
try:
    for i, entity in enumerate(model.dictionary.entities()):   # 加括号调用
        if i >= 20:
            break
        print(entity.title)
except Exception as e:
    print("枚举实体名失败:", e)
    # 看看 Dictionary 到底有哪些属性
    print([x for x in dir(model.dictionary) if not x.startswith('_')])
