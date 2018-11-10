"""
Serena Zhang
Nov 5th
"""

"""
This script calls functions from modules and completes a whole run for the google amazon dataset
"""

## Preprocess any specific columns (e.g. price column for differnt currency)
## Then get the three types of matrices for both google and amazon data (6 matrices in total)
## Call similarity functions on 3 pairs of matrices
## Concatenate previous results to form the final dataset for modeling
## Call modeling functions (train test split etc)

import pandas as pd
import sys
import numpy as np
sys.path.append('..')
from modules.preprocessing import Preprocessing
from modules.preprocessing.generate_labels import gen_labels
from modules.feature_generation.gen_similarities import similarities
from sklearn.model_selection import train_test_split

df1 = pd.read_csv("data/amazon_sample.csv")
df2 = pd.read_csv("data/google_sample.csv")
match = pd.read_csv("data/amazon_google_sample_match.csv")
#df1 = pd.read_csv('/Users/shihhuayu/capstone/companies_data_neoway_subsample/reference.csv')
#df2 = pd.read_csv('/Users/shihhuayu/capstone/companies_data_neoway_subsample/input.csv')

'''
custom data cleaning 
#we still need to convert currency. right now just ignoring currency effect
'''

#df2["price"] = df2.price.str.replace(r"[a-zA-Z]",'').astype(float)

'''
preprocess both dataframes
'''
processed_data = Preprocessing().overall_preprocess(df1,df2) #may take a while bc loading pretrained word embedding model

'''
get numerical data
'''
num_matrix_1, num_matrix_2 = processed_data[1]["numerical"][0], processed_data[1]["numerical"][1]
embed_matrix_1, embed_matrix_2 = processed_data[1]["word_embedding_fields"][0],processed_data[1]["word_embedding_fields"][1]

'''
calculate similarities
'''
num_final_data = similarities().numerical_similarity_on_matrix(num_matrix_1, num_matrix_2)
embed_final_data = similarities().vector_similarity_on_matrix(embed_matrix_1, embed_matrix_2)

'''
concatenate all data
'''

'''
train test split
'''
y = gen_labels(df1['id'], df2['id'], match)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, stratify = y) #** NEEDS TESTING **


'''
modeling
'''

