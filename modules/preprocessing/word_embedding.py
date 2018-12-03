import numpy as np
import gensim
import nltk
from .process_text import Process_text
from collections import defaultdict
import math
from sklearn.feature_extraction.text import TfidfVectorizer


class MeanEmbeddingVectorizer():
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = word2vec.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0) for words in X])

class MinEmbeddingVectorizer():
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = word2vec.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([np.nanmin([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0) for words in X])

class MaxEmbeddingVectorizer():
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = word2vec.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([np.nanmax([self.word2vec[w] for w in words if w in self.word2vec]
                                or [np.zeros(self.dim)], axis=0) for words in X])


class TfidfEmbeddingVectorizer():
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = word2vec.vector_size

    def fit(self, X, y=None):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0) for words in X])


class Word_embedding():
    def __init__(self,word_embedding_model,path_to_pretrained_model):
        """
        Initialize model variable; type: gensim.models.keyedvectors.Word2VecKeyedVectors
        """
        #loading may take a while
        if word_embedding_model == "word2vec":
            self.model = gensim.models.KeyedVectors.load_word2vec_format(path_to_pretrained_model, binary=True)
        elif word_embedding_model == "fasttext":
            self.model = gensim.models.FastText.load_fasttext_format(path_to_pretrained_model)
        elif word_embedding_model == "glove":
            self.model = gensim.models.KeyedVectors.load_word2vec_format(path_to_pretrained_model)
    

    def dataframe_to_embedding(self,df,attribute_list, weight = 'tfidf'):

        """
        Extract word embeddings from original dataset
        :arg:
            df: pd dataframe of the dataset
            attribute_list: list of attribute names (in string type) relevant for word embeddings
        :return:
            np.array of shape (# of entities, # of attributes, dim of word embedding)
        """

        if type(attribute_list[0]) == int:
            new = []
            for i in attribute_list:
                new.append(list(df.columns)[i])
            attribute_list = new
        else:
            if bool(set(attribute_list) - set(
                    df.columns.values)) == True:  # check if input attributes exist
                raise ValueError('Attributes provided do not exist.')

        #extract relevant columns
        X_transformed = []
        for attribute in attribute_list:
            X = df[attribute].apply(str).apply(tokenize_normalize_sentence).tolist()
            if weight == 'tfidf':
                embed = TfidfEmbeddingVectorizer(self.model) #using tf-idf
            elif weight == 'mean':
                embed = MeanEmbeddingVectorizer(self.model)
            elif weight == 'min':
                embed = MinEmbeddingVectorizer(self.model)
            elif weight == 'max':
                embed = MaxEmbeddingVectorizer(self.model)

            embed.fit(X)
            X_transformed += [embed.transform(X)]

        return np.swapaxes(np.vstack([X_transformed]),0,1)



class Word_embedding_new():
    def __init__(self,word_embedding_model,path_to_pretrained_model):
        """
        Initialize model variable; type: gensim.models.keyedvectors.Word2VecKeyedVectors
        """
        #loading may take a while
        if word_embedding_model == "word2vec":
            self.model = gensim.models.KeyedVectors.load_word2vec_format(path_to_pretrained_model, binary=True)
        elif word_embedding_model == "fasttext":
            self.model = gensim.models.FastText.load_fasttext_format(path_to_pretrained_model)
        elif word_embedding_model == "glove":
            self.model = gensim.models.KeyedVectors.load_word2vec_format(path_to_pretrained_model)

    
    def fit_embedding(self,df,attribute_list,weight = 'tfidf'):
    
        if type(attribute_list[0]) == int:
            new = []
            for i in attribute_list:
                new.append(list(df.columns)[i])
            attribute_list = new
    
        else:
            if bool(set(attribute_list) - set(
                    df.columns.values)) == True:  # check if input attributes exist
                raise ValueError('Attributes provided do not exist.')

        embed = [] #save embeddings into a list
        for attribute in attribute_list:
            X = df[attribute].apply(str).apply(tokenize_normalize_sentence).tolist()
            if weight == 'tfidf':
                ev = TfidfEmbeddingVectorizer(self.model) #using tf-idf
            elif weight == 'mean':
                ev = MeanEmbeddingVectorizer(self.model)
            elif weight == 'min':
                ev = MinEmbeddingVectorizer(self.model)
            elif weight == 'max':
                ev = MaxEmbeddingVectorizer(self.model)
            
            embed += [ev.fit(X)]
        return embed, attribute_list


def df_to_embedding(embed_attribute_list,row):
    
    embed_list = embed_attribute_list[0]
    attr_list = embed_attribute_list[1]
    
    row_transformed = []
    for i in range(len(embed_list)):
        normalized_sentence = row[attr_list[i]].apply(str).apply(tokenize_normalize_sentence).tolist()
        row_transformed += [embed_list[i].transform(normalized_sentence)]

    return np.swapaxes(np.vstack([row_transformed]),0,1)


def tokenize_normalize_sentence(sentence):
    """
    tokenize and normalize a sentence
    :arg:
        sentence: sentence to convert to list of normalized words; type: string
    :return:
        list of strings
    """
    processed_sentence = Process_text().standard_text_normalization(sentence)
    processed_sentence = nltk.word_tokenize(processed_sentence)

    return processed_sentence





