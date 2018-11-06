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
sys.path.append('..')
from modules.preprocessing import Preprocessing
from modules.feature_generation.gen_similarities import similarities
df1 = pd.read_csv("data/amazon_sample.csv")
df2 = pd.read_csv("data/google_sample.csv")

'''
custom data cleaning 
#we still need to convert currency. right now just ignoring currency effect
'''

df2["price"] = df2.price.str.replace(r"[a-zA-Z]",'').astype(float)

'''
get numerical data
'''

# get names of numerical columns
num_cols_df1,num_cols_df2 = Preprocessing().overall_preprocess(df1,df2)

# convert to numpy matrix and calculate similarities for each cross product of samples
num_matrix_1 = df1[num_cols_df1].values
num_matrix_2 = df2[num_cols_df2].values

'''
calculate similarities
'''

num_final_data = similarities().numerical_similarity_on_matrix(num_matrix_1,num_matrix_2)


'''
concatenate all data
'''

'''
train test split
'''

'''
modeling
'''

