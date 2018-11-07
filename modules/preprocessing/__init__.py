import numpy as np
import sys
from .word_embedding import Word_embedding
from .process_text import Process_text
#sys.path.append('..')

class Preprocessing():
    def __init__(self):
        pass
    def process_zipcode(self):
        pass
    def process_phone_num(self):
        pass


    def overall_preprocess(self,df1,df2):
        """
        This function divides the given raw data into three preprocessed sub-dataset (or numpy matrices):
        - numerical matrix
        - special treatment columns
        - word embedding matrix; shape: (# of attributes, # of entities, dim of word embedding(e.g. 300))

        :arg: df1: reference df; df2: input df
        :return: three matrices
        """
        divide_col = {"numerical_cols": [],"special_field_cols":[],"word_embedding_cols":[["name","addressStreet","addressCity","addressState"],["name","addressStreet","addressCity","addressState"]]}
        #TODO: write function "divide_columns" that returns divided column names (ie numerical, special, word embedding columns)
        #and call the function here and save to "divide_col"
        #so that divide thats in the value returned from function "divide_columns"
        
        df1_num_col = df1.select_dtypes(include=[np.number]).columns.tolist() #TODO: move to function "divide_columns", and remove "serial" from col list
        df2_num_col = df2.select_dtypes(include=[np.number]).columns.tolist() #TODO: move to function "divide_columns", and remove "serial" from col list
        divide_col["numerical_cols"].append(df1_num_col) #TODO: move to function "divide_columns"
        divide_col["numerical_cols"].append(df2_num_col) #TODO: move to function "divide_columns"
        


        #process word embeddings
        embed = None
        df1_embed = None
        df2_embed = None
        
        if divide_col["word_embedding_cols"][0] and divide_col["word_embedding_cols"][1]: #process only if both col lists are not empty
            embed = Word_embedding('/Users/shihhuayu/capstone/GoogleNews-vectors-negative300.bin') #initialization may take a while
            df1_embed = embed.dataframe_to_embedding(df1,divide_col["word_embedding_cols"][0])
            df2_embed = embed.dataframe_to_embedding(df2,divide_col["word_embedding_cols"][1])
        else: #else set to empty arrays
            df1_embed = np.array([])
            df2_embed = np.array([])

        ## after finishing preprocessing
        processed_data = {"numerical":[df1[divide_col["numerical_cols"][0]].values,
                                       df2[divide_col["numerical_cols"][1]].values],
                          "special_fields":[],
                          "word_embedding_fields":[df1_embed,df2_embed]}
                          
        return processed_data
