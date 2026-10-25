from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import gensim.models as g
import re


def sentence_preprocessing(sentence):
    '''
    Performs a stop words removal and lemmatization on a sentence
    Input :
        sentence : sentence to preprocess
        path : path in which the RDF graph will be saved
        threshold : pageRank threshold to use
    Output :
        RDF graph
    '''

    # lemmatization
    lemmatizer = WordNetLemmatizer()

    # stop words removal
    stop_words = set(stopwords.words('english'))
    output = []
    sentence = re.sub(r'[^\w\s]','',sentence)
    sentence = sentence.split()
    for word in sentence:
        if word not in stop_words:
            output.append(lemmatizer.lemmatize(word))
            
    return output


def tagged_document(list_of_list_of_words):
   for i, list_of_words in enumerate(list_of_list_of_words):
      yield g.doc2vec.TaggedDocument(sentence_preprocessing(list_of_words), [i])


def doc2vec_embeddings(sentences):
    model = g.doc2vec.Doc2Vec(vector_size=600, epochs=10)
    model.build_vocab(list(tagged_document(sentences)))
    return [model.infer_vector(sentence_preprocessing(s)) for s in sentences]

