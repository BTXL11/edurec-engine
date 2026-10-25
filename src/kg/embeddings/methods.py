from .doc2vec import doc2vec_embeddings
from .embedder import embedder_embeddings
from .roberta import roberta_embeddings
from .tfidf import tfidf_embeddings
from .bert import bert_embeddings
from .gpt2 import gpt2_embeddings
import pandas as pd

import pandas as pd


def embeddings(sentences, resources, methods, save = False, path = ''):

    embeddings_df = pd.DataFrame()

    if 'EMBEDD-ER' in methods:
        embeddings_df['EMBEDD-ER'] = embedder_embeddings(resources)

    if 'BERT' in methods:
        embeddings_df['BERT'] = bert_embeddings(sentences)

    if 'Doc2Vec' in  methods:
        embeddings_df['Doc2Vec'] = doc2vec_embeddings(sentences)

    if 'TF-IDF' in methods:
        embeddings_df['TF-IDF'] = tfidf_embeddings(sentences)

    if 'GPT2' in methods:
        embeddings_df['GPT2'] = gpt2_embeddings(sentences)

    if 'ROBERTA' in methods:
        embeddings_df['ROBERTA'] = roberta_embeddings(sentences)
    
    if save:
        embeddings_df.to_csv(path+"/embeddings.csv", sep = '|')

    return embeddings_df
