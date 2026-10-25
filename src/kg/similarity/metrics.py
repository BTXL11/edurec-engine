import numpy as np


def norm(v1, v2, n):
    '''
    Calculates the 'n' norm between 2 vectors
    Input :
        v1, v2 : input vectors
        n : norm to use
    Output :
        norm value
    '''
    v1 = np.array(v1)
    v2 = np.array(v2)
    assert len(v1) == len(v2)
    return np.linalg.norm(v1 - v2, n)


def cosine_similarity(v1, v2):
    '''
    Calculates the cosine similarity between two vectors
    Input :  
        v1, v2 : the input vectors
    Output :
        cosine similarity score
    '''
    return np.dot(v1, v2) / (np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2))


def distance_matrix(vectors, method = 'cosine', n = 2):
    '''
    Calculates the 'n' norm between a set of vectors
    Input :
        vectors : input vectors
        method : similarity method to use
        n : norm to use if method == norm
    Output :
        distance matrix
    '''
    l = len(vectors)
    matrix = np.zeros((l,l))
    for i in range(l):
        for j in range(i ,l):
            if method == 'cosine':
                value = cosine_similarity(vectors[i], vectors[j])
            elif method == 'norm':
                value = norm(vectors[i], vectors[j], n)
            else :
                value = 0
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def jaccard_index(a, b):
    '''
    Calculates the Jaccard index between two sets or lists
    Input :
        a,b : input sets or lists
    Output :
        jaccard index
    '''
    if(type(a) == list):
        a = set(a)
    if(type(b) == list):
        b = set(b)
    return len(a & b) / len(a | b)


def jaccard_matrix(concepts):
    '''
    Calculates the Jaccard index between a list of sets or lists
    Input :
        concepts : input list of sets or lists
    Output :
        jaccard index matrix
    '''
    l = len(concepts)
    matrix = np.zeros((l,l))
    for i in range(l):
        for j in range(i ,l):
            matrix[i, j] = jaccard_index(concepts[i], concepts[j])
            matrix[j, i] = jaccard_index(concepts[i], concepts[j])
    return matrix