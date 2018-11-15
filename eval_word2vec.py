import gensim
import pandas as pd
import nltk
import string
# from modules.preprocessing.process_text import Process_text

path_to_pretrained_model = 'data/embeddings/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(path_to_pretrained_model, binary=True)

df1 = pd.read_csv("data/amazon_google/sample/amazon_sample.csv")
df2 = pd.read_csv("data/amazon_google/sample/google_sample.csv")


for col in ['title', 'description', 'manufacturer']:
    all_words = set()
    # df1['description'].str.lower().str.split().apply(all_words.update)
    if col=='title':
        col2='name'
    else:
        col2=col
    df1[col].str.lower().str.replace('[{}]'.format(string.punctuation), '').str.replace(r'[0-9]+', '').str.split().apply(all_words.update)
    df2[col2].astype('str').str.lower().str.replace('[{}]'.format(string.punctuation), '').str.replace(r'[0-9]+', '').str.split().apply(all_words.update)

    absent = [word for word in all_words if word not in model]
    print("{a} are missing out of {b} in {colm}".format(a=len(absent),
                                          b=len(all_words),
                                                       colm=col))
# def sentence_to_embedding(sentence):
#     """
#     Extract word embeddings from a sentence
#     :arg:
#         sentence: sentence to convert to word embeddings; type: string
#     :return:
#         np.array of shape (dim of word embedding,)
#     """
#     text_processor = Process_text()
#     processed_sentence = nltk.word_tokenize(sentence)
#     processed_sentence = text_processor.remove_non_ascii(processed_sentence)
#     processed_sentence = text_processor.to_lowercase(processed_sentence)
#     processed_sentence = text_processor.remove_punctuation(processed_sentence)
#     # processed_sentence = text_processor.replace_numbers(processed_sentence) #TODO: try
#     processed_sentence = text_processor.remove_stopwords(processed_sentence)
#
#     return ''.join(processed_sentence)