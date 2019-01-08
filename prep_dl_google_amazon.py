import sys
sys.path.append('..')
sys.path.append('/anaconda/lib/python3.6/site-packages')
from modules.preprocessing import Preprocessing
from modules.preprocessing.generate_labels import gen_labels
from modules.feature_generation.gen_similarities import similarities
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from modules.preprocessing.process_text import Process_text


def tokenize_normalize_sentence(sentence):
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

batch_size = 100

df1 = pd.read_csv("data/amazon_google/sample/amazon_sample_v1.csv")
df2 = pd.read_csv("data/amazon_google/sample/google_sample_v1.csv")
match_df = pd.read_csv("data/amazon_google/sample/amazon_google_sample_match_v1.csv")

df1['description'] = df1['description'].apply(str).apply(tokenize_normalize_sentence)
df2['description'] = df2['description'].apply(str).apply(tokenize_normalize_sentence)

all_words = set(x for lst in df1['description'] for x in lst)
all_words = all_words.union(set(x for lst in df2['description'] for x in lst))

word2idx = {}
for i, word in enumerate(all_words):
    if word not in word2idx.keys():
        word2idx[word] = i

print('total # of words', len(word2idx))

df1_id = 'id'
df2_id = 'id'
match_id1 = 'idAmazon'  #corresponds to df1_id
match_id2 = 'idGoogleBase'  #corresponds to df2_id
df2["price"] = df2.price.str.replace(r"[a-zA-Z]",'').astype(float)

# Save for later use to generate labels
df1_id_col = df1[df1_id]
df2_id_col = df2[df2_id]

# Drop id columns because we don't need to compute id similarity
df1 = df1.drop(columns=[df1_id])
df2 = df2.drop(columns=[df2_id])

processed_data = Preprocessing().overall_preprocess(
    df1.drop(columns=['description']), df2.drop(columns=['description']),
    special_columns=['title', 'manufacturer'],
    word_embedding_model='none') # may take a while bc loading pretrained word embedding model

num_matrix_1, num_matrix_2 = processed_data["numerical"][0], processed_data["numerical"][1]
spc_matrix_1, spc_matrix_2 = processed_data["special_fields"][0], processed_data["special_fields"][1]
num_final_data = similarities().numerical_similarity_on_matrix(num_matrix_1,num_matrix_2)
spc_final_data_0 = similarities().text_similarity_on_matrix(spc_matrix_1,spc_matrix_2,method='jaccard')
spc_final_data_1 = similarities().text_similarity_on_matrix(spc_matrix_1,spc_matrix_2,method='lavenshtein')
spc_final_data_2 = similarities().text_similarity_on_matrix(spc_matrix_1,spc_matrix_2,method='jaro_winkler')

df1['key'] = 0
df2['key'] = 0
merged = pd.merge(df1, df2, on='key')[['description_x', 'description_y']]

'''
train-test split
'''
non_empty = []

for m in num_final_data, spc_final_data_0, spc_final_data_1, spc_final_data_2:
    if m.size !=0:
        non_empty.append(m)

sim = np.concatenate([i for i in non_empty], axis = 1)
y = gen_labels(df1_id_col, df2_id_col, match_df, 'idAmazon', 'idGoogleBase')


from sklearn.model_selection import train_test_split
sim_train, sim_val, y_train, y_val, desc_train, desc_val = train_test_split(
                     sim,
                     y, merged,
                     test_size=0.33,
                     stratify=y,
                     random_state=42)
print(sum(y_train), sum(y_val))
print(len(y_train), len(y_val))
sim_dev, sim_test, y_dev, y_test, desc_dev, desc_test = train_test_split(
                     sim_val,
                     y_val, desc_val,
                     test_size=0.6,
                     stratify=y_val,
                     random_state=42)
print("test", sum(y_test),"val", sum(y_dev))
print("len test",len(y_test), "len val", len(y_dev))

'''
Embeddings
'''

def load_embeddings(word2idx, glove_file):
    corpus_words = set()
    for key in word2idx.keys():
        # if key not in {'<NULL>','<UNK>'}:
        corpus_words.add(key)

    glove_big = {}
    with open(glove_file, "rb") as infile:
        for line in infile:
            parts = line.split()
            word = parts[0].decode()  # .lower()
            nums = map(float, parts[1:])
            if (word in corpus_words):
                glove_big[word] = list(nums)

    idx2word = {v: k for k, v in word2idx.items()}

    weights_matrix = np.random.normal(scale=0.6, size=(
    len(idx2word), 300))  # np.zeros((len(idx2word), 300))
    words_found = 0

    for word in corpus_words:
        if word in glove_big.keys():
            weights_matrix[word2idx[word]] = glove_big[word]
            words_found += 1
    print("%d words found out of %d" % (words_found, len(idx2word)))

    return weights_matrix


weights_matrix = load_embeddings(word2idx, '/Users/serenazhang/Desktop/capstone/capstone-entitymatching/glove/glove.840B.300d.txt')


import json
with open("data/amazon_google/dl_data/word2idx_v1", "w") as fp:
    json.dump(word2idx , fp)

# Save preprocessed data
np.save("data/amazon_google/dl_data/glove_weights_v1", weights_matrix)
np.save("data/amazon_google/dl_data/description_train", desc_train)
np.save("data/amazon_google/dl_data/sim_train", sim_train)
np.save("data/amazon_google/dl_data/targets_train", y_train)

np.save("data/amazon_google/dl_data/description_val", desc_val)
np.save("data/amazon_google/dl_data/sim_val", sim_val)
np.save("data/amazon_google/dl_data/targets_val", y_val)

np.save("data/amazon_google/dl_data/description_test", desc_test)
np.save("data/amazon_google/dl_data/sim_test", sim_test)
np.save("data/amazon_google/dl_data/targets_test", y_test)
