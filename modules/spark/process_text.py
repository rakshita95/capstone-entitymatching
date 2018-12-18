import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import re

class Process_text():
    def __init__(self):
        pass

    def remove_non_ascii(self,words):
        """Convert non-ASCII characters to ASCII (remove umlauts, accents etc.) from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def to_lowercase(self,words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def remove_punctuation(self,words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words
    
    def remove_nan(self,words):
        """remove nan from list of tokenized words"""
        new_words = []
        for word in words:
            if word != 'nan':
                new_words.append(word)
        return new_words

    def standard_text_normalization(self,text):
        """Normalize text
           :arg: string type
        """
        new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_text = new_text.lower()
        new_text = re.sub(r'[^\w\s]', '', new_text)
        new_text = re.sub('nan','', new_text) #replace 'nan' as empty string
        
        return new_text


#    def replace_numbers(self,words):
#        """Replace all interger occurrences in list of tokenized words with textual representation"""
#        p = inflect.engine()
#        new_words = []
#        for word in words:
#            if word.isdigit():
#                new_word = p.number_to_words(word)
#                new_words.append(new_word)
#            else:
#                new_words.append(word)
#        return new_words

    def remove_stopwords(self,words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words

#    def stem_words(self,words):
#        """Stem words in list of tokenized words"""
#        stemmer = LancasterStemmer()
#        stems = []
#        for word in words:
#            stem = stemmer.stem(word)
#            stems.append(stem)
#        return stems

#    def lemmatize_verbs(self,words):
#        """Lemmatize verbs in list of tokenized words"""
#        lemmatizer = WordNetLemmatizer()
#        lemmas = []
#        for word in words:
#            lemma = lemmatizer.lemmatize(word, pos='v')
#            lemmas.append(lemma)
#        return lemmas



