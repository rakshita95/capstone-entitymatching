import numpy as np
import sys
from .word_embedding import Word_embedding
from .process_text import Process_text
from preprocess_special_columns import *
#sys.path.append('..')

def divide_columns(df, special_columns):
    """
    Returns the indices of the numeric, word embedding and special value columns
    :param df1:
    :param special_columns:
    :return:
    """
    t = 0
    embeddings = []
    numeric = []
    special = []

    if type(special_columns[1]) == str:
        for s in special_columns:
            t = list(df.columns).index(s)
            special.append(t)
    else:
        special = special_columns

    for i in df.loc[0].tolist():
        if type(i) == str:
            embeddings.append(t)
        elif type(i) == float or type(i) == int:
            numeric.append(t)
        t += 1
    return numeric, special, embeddings


class Preprocessing():
    def __init__(self):
        pass
    def process_zipcode(self):
        pass
    def process_phone_num(self):
        pass


    def overall_preprocess(self,df1,df2,
                           special_columns,
                           phone_number,
                           path = '/Users/shihhuayu/capstone/GoogleNews-vectors-negative300.bin'):
        """
        This function divides the given raw data into three preprocessed sub-dataset (or numpy matrices):
        - numerical matrix
        - special treatment columns
        - word embedding matrix; shape: (# of attributes, # of entities, dim of word embedding(e.g. 300))

        :arg: df1: reference df; df2: input df; special_columns: a list of
                indices or labels of the columns containing special information
                such as email, address, phone number, name;
                path: path for the word embedding dictionary;
                phone: give the phone number as special field
        :return: three matrices
        """
        divide_col = {"numerical_cols": [],
                      "special_field_cols":[],
                      "word_embedding_cols":[["name","addressStreet","addressCity","addressState"],
                                             ["name","addressStreet","addressCity","addressState"]]}
        #TODO: write function "divide_columns" that returns divided column names (ie numerical, special, word embedding columns)
        #and call the function here and save to "divide_col"
        #so that divide thats in the value returned from function "divide_columns"
        
        #df1_num_col = df1.select_dtypes(include=[np.number]).columns.tolist() #TODO: move to function "divide_columns", and remove "serial" from col list
        #df2_num_col = df2.select_dtypes(include=[np.number]).columns.tolist() #TODO: move to function "divide_columns", and remove "serial" from col list
        #divide_col["numerical_cols"].append(df1_num_col) #TODO: move to function "divide_columns"
        #divide_col["numerical_cols"].append(df2_num_col) #TODO: move to function "divide_columns"

        n, s, w = divide_columns(df1, special_columns)
        divide_col['numerical_cols'] = n
        divide_col['special_field_cols'] = s
        divide_col['word_embedding_cols'] = w



        #process word embeddings
        if divide_col["word_embedding_cols"]: #process only if both col lists are not empty
            embed = Word_embedding(path) #initialization may take a while
            df1_embed = embed.dataframe_to_embedding(df1,divide_col["word_embedding_cols"])
            df2_embed = embed.dataframe_to_embedding(df2,divide_col["word_embedding_cols"])
        else:
            df1_embed = np.array([])
            df2_embed = np.array([])

        # process special columns
        if divide_col['special_field_cols']:
            df1_special = preprocess_special_fields(df1.iloc[:,
                                                    divide_col['special_field_cols']],
                                                    phone_number)
            df2_special = preprocess_special_fields(df2.iloc[:,
                                                    divide_col['special_field_cols']],
                                                    phone_number)
        else:
            df1_special = np.array([])
            df2_special = np.array([])

        # process numeric columns
        if divide_col['numerical_cols']:
            df1_numeric = df1.iloc[:, divide_col['numerical_cols']]
            df2_numeric = df2.iloc[:, divide_col['numerical_cols']]
        else:
            df1_numeric = np.array([])
            df2_numeric = np.array([])

        ## after finishing preprocessing
        processed_data = {"numerical":[df1_numeric, df2_numeric],
                          "special_fields":[df1_special, df2_special],
                          "word_embedding_fields":[df1_embed, df2_embed]
                          }
        return processed_data
