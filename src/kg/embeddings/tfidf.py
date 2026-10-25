from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE


def tfidf_embeddings(sentences, reduce = True):
    vectorizer = TfidfVectorizer(stop_words="english")
    sentences_vector_tfidf = vectorizer.fit_transform(sentences).A
    if reduce:
        sentences_vector_tfidf = reduce_embeddings(sentences_vector_tfidf)
    sentences_vector_tfidf = [list(k) for k in sentences_vector_tfidf]
    return sentences_vector_tfidf


def reduce_embeddings(X):
    return TSNE(n_components = 700, learning_rate = 10, init = 'random', perplexity = 5, n_iter = 10000, method = 'exact').fit_transform(X)
