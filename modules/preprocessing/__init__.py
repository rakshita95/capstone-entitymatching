import numpy as np
import sys
from .word_embedding import Word_embedding
from .process_text import Process_text
from .preprocess_special_columns import *
from .process_text import Process_text
#sys.path.append('..')

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def divide_columns(df, special_columns=[]):
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

    if special_columns:
        if type(special_columns[0]) == str:
            for s in special_columns:
                t = list(df.columns).index(s)
                special.append(t)
        else:
            special = special_columns
    else:
        special = []

    t = 0

    for i in df.loc[0].tolist():
        if type(i) == str:
            embeddings.append(t)
        #elif type(i) in [int, float, np.int64, np.float32]:
        elif is_number(i):
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
                           special_columns=[],
                           phone_number=[],
                           address_columns = [],
                           geocode_address=False,
                           api_key=None,
                           path='data/embeddings/GoogleNews-vectors-negative300.bin'):

        """
        This function divides the given raw data into three preprocessed sub-dataset (or numpy matrices):
        - numerical matrix
        - special treatment columns
        - word embedding matrix; shape: (# of attributes, # of entities, dim of word embedding(e.g. 300))
        :param df1: pd.df
        :param df2: pd.df
        :param special_columns: a list of special columns values
        :param phone_number:
        :param address_columns:
        :param geocode_address:
        :param api_key:
        :param path:
        :return:
        """

        s = set(special_columns)
        s.update(address_columns)
        s.update(phone_number)
        special_columns = list(s)

        divide_col = {"numerical_cols": [],
                      "special_field_cols":[],
                      "word_embedding_cols":[]}


        n, s, w = divide_columns(df1, special_columns)
        divide_col['numerical_cols'] = n
        divide_col['special_field_cols'] = s
        divide_col['word_embedding_cols'] = w

        print('**** df1 divide columns ****')
        [print(i, ': ', df1.columns[j].values) for i, j in divide_col.items()]

        print('\n','**** df2 divide columns ****')
        [print(i, ': ', df2.columns[j].values) for i, j in divide_col.items()]

        divide_col['word_embedding_cols'] = []
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

            df1_special, lat1,long1 = preprocess_special_fields(df1.iloc[:,
                                                    divide_col['special_field_cols']],
                                                    phone_number,
                                                    address_columns,
                                                    geocode_address,
                                                    api_key)
            df2_special, lat2,long2 = preprocess_special_fields(df2.iloc[:,
                                                    divide_col['special_field_cols']],
                                                    phone_number,
                                                    address_columns,
                                                    geocode_address,
                                                    api_key)

            if geocode_address and api_key:
                df1['lat'] = lat1
                df1['long'] = long1
                df2['lat'] = lat2
                df2['long'] = long2
                divide_col['numerical_cols'] = divide_col['numerical_cols'] +\
                                               [-2,-1]

        else:
            df1_special = np.array([])
            df2_special = np.array([])

        # process numeric columns
        if divide_col['numerical_cols']:
            df1_numeric = df1.iloc[:, divide_col['numerical_cols']].as_matrix()
            df2_numeric = df2.iloc[:, divide_col['numerical_cols']].as_matrix()

        else:
            df1_numeric = np.array([])
            df2_numeric = np.array([])

        ## after finishing preprocessing
        processed_data = {"numerical":[df1_numeric, df2_numeric],
                          "special_fields":[df1_special, df2_special],
                          "word_embedding_fields":[df1_embed, df2_embed]
                          }
        return processed_data
