# from gensim.models import KeyedVectors
# import os
#
# src = "models/wikipedia/zhwiki_20180420_300d.txt"
# dst = "models/wikipedia/zhwiki_20180420_300d.kv"
#
# if os.path.exists(dst):
#     print("二进制文件已存在，跳过转换")
# else:
#     print("首次加载 txt，需要 20~60 分钟，请耐心等待...")
#     model = KeyedVectors.load_word2vec_format(src, binary=False)
#     print("加载完成，正在保存为二进制格式...")
#     model.save(dst)
#     print(f"已保存到 {dst}")
path = "models/wikipedia/zhwiki_20180420_300d.txt"
count = 0
with open(path, "rb") as f:
    for line in f:
        count += 1
print(f"实际行数: {count}")
print(f"声明词数: 1572706 (数据行应为 1572707)")
print(f"差距: {count - 1572707}")