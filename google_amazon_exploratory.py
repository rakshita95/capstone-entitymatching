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

#************************* USER INPUT **********************************************
'''
read data
'''
df1 = pd.read_csv("samples/acm_dblp/acm_sample.csv")
df2 = pd.read_csv("samples/acm_dblp/dblp_sample.csv")
match_df = pd.read_csv("samples/acm_dblp/acm_dblp_sample_match.csv")

'''
specify id names
'''
df1_id = 'id'
df2_id = 'id'
match_id1 = 'idACM' # corresponds to df1_id
match_id2 = 'idDBLP' # corresponds to df2_id

'''
custom data cleaning, currently this section is for google dataset only
we still need to convert currency. right now just ignoring currency effect
'''
#df2["price"] = df2.price.str.replace(r"[a-zA-Z]",'').astype(float)

#***********************************************************************************

'''
id column manipulation
'''
# save for later use to generate labels
df1_id_col = df1[df1_id]
df2_id_col = df2[df2_id]

# drop id columns because we don't need to compute id similarity
df1 = df1.drop(columns = [df1_id])
df2 = df2.drop(columns = [df2_id])

'''
preprocess both dataframes
'''
processed_data = Preprocessing().overall_preprocess(df1,df2) #may take a while bc loading pretrained word embedding model

'''
get numerical data
'''
num_matrix_1,num_matrix_2 = processed_data["numerical"][0],processed_data["numerical"][1]
embed_matrix_1,embed_matrix_2 = processed_data["word_embedding_fields"][0],processed_data["word_embedding_fields"][1]
spc_matrix_1,spc_matrix_2 = processed_data["special_fields"][0],processed_data["special_fields"][1]

'''
calculate similarities
'''
num_final_data = similarities().numerical_similarity_on_matrix(num_matrix_1,num_matrix_2)
embed_final_data = similarities().vector_similarity_on_matrix(embed_matrix_1,embed_matrix_2)
spc_final_data = similarities().text_similarity_on_matrix(spc_matrix_1,spc_matrix_2)

'''
concatenate all data
'''
# only concatenate non-empty similarity matrices
non_empty = []

for m in num_final_data, embed_final_data, spc_final_data:
    if m.size !=0:
        non_empty.append(m)

x = np.concatenate([i for i in non_empty], axis = 1)

print(x.shape)

'''
train test split
'''
# generate y labels
y = gen_labels(df1_id_col, df2_id_col, match_df, match_id1, match_id2)

# simple check to see if x and y match in size
print (y.shape[0] == x.shape[0])
print(y.sum() == match_df.shape[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, stratify = y)

'''
modeling
'''

