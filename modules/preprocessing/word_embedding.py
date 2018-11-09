import numpy as np
import gensim
import nltk
from .process_text import Process_text

class Word_embedding():
    def __init__(self,path_to_pretrained_model):
        """
        Initialize model variable; type: gensim.models.keyedvectors.Word2VecKeyedVectors
        """
        self.model = gensim.models.KeyedVectors.load_word2vec_format(path_to_pretrained_model, binary=True)
        #loading may take a while

    def sentence_to_embedding(self,sentence):
        """
        Extract word embeddings from a sentence
        :arg:
            sentence: sentence to convert to word embeddings; type: string
        :return:
            np.array of shape (dim of word embedding,)
        """
        text_processor = Process_text()
        processed_sentence = nltk.word_tokenize(sentence)
        processed_sentence = text_processor.remove_non_ascii(processed_sentence)
        processed_sentence = text_processor.to_lowercase(processed_sentence)
        processed_sentence = text_processor.remove_punctuation(processed_sentence)
        #processed_sentence = text_processor.replace_numbers(processed_sentence) #TODO: try
        #processed_sentence = text_processor.remove_stopwords(processed_sentence) #TODO: try
        #processed_sentence = text_processor.stem_words(processed_sentence) #TODO: try this or lemmatize_verbs
        #processed_sentence = text_processor.lemmatize_verbs(processed_sentence) #TODO: try this or stem_words
    
        #now using simple average (TODO: tf-idf version)
        dim = self.model.vector_size
        return np.mean([self.model[w] for w in sentence if w in self.model] or [np.zeros(dim)], axis=0)

    def dataframe_to_embedding(self,df,attribute_list):
        """
        Extract word embeddings from original dataset
        :arg:
            df: pd dataframe of the dataset
            attribute_list: list of attribute names (in string type) relevant for word embeddings
        :return:
            np.array of shape (# of attributes, # of entities, dim of word embedding)
        """
        if bool(set(attribute_list)-set(df.columns.values))==True: #check if input attributes exist
            raise ValueError('Attributes provided do not exist.')
        if type(attribute_list[0]) == int:
            new = []
            for i in attribute_list:
                new.append(list(df.columns)[i])
            attribute_list = new

        else:
            return np.swapaxes(np.vstack([[np.vstack(df[attribute].apply(str).apply(self.sentence_to_embedding))] for attribute in attribute_list]),0,1)


