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
    def __init__(self,path_to_pretrained_model):
        """
        Initialize model variable; type: gensim.models.keyedvectors.Word2VecKeyedVectors
        """
        self.model = gensim.models.KeyedVectors.load_word2vec_format(path_to_pretrained_model, binary=True)
        #loading may take a while
        #self.model = gensim.models.FastText.load_fasttext_format(path_to_pretrained_model)
        
    def tokenize_normalize_sentence(self,sentence):
        """
        tokenize and normalize a sentence
        :arg:
            sentence: sentence to convert to list of normalized words; type: string
        :return:
            list of strings
        """
        text_processor = Process_text()
        processed_sentence = nltk.word_tokenize(sentence)
        processed_sentence = text_processor.remove_non_ascii(processed_sentence)
        processed_sentence = text_processor.to_lowercase(processed_sentence)
        processed_sentence = text_processor.remove_punctuation(processed_sentence)
        processed_sentence = text_processor.remove_nan(processed_sentence)
        processed_sentence = text_processor.remove_stopwords(processed_sentence)
        
        return processed_sentence
    

    def dataframe_to_embedding(self,df,attribute_list):

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
            X = df[attribute].apply(str).apply(self.tokenize_normalize_sentence).tolist()
            embed = TfidfEmbeddingVectorizer(self.model) #using tf-idf
            embed.fit(X)
            X_transformed += [embed.transform(X)]

        return np.swapaxes(np.vstack([X_transformed]),0,1)





