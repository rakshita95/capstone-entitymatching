import numpy as np
import sys
from .word_embedding import Word_embedding
from .word_embedding import Word_embedding_new
from .word_embedding import df_to_embedding
from .process_text import Process_text
from .preprocess_special_columns import *
from .process_text import Process_text
#sys.path.append('..')

def is_number(s):

    try: #nan values would pass but cannot use re.sub
        if s != None:
            s = re.sub(r'[^\w\s]', '', s) #remove punctuation; doesnt remove all special char, eg _"

    except TypeError:
        pass

    try:
    
        float(s) #float('nan'),float('NaN'), etc exist
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
    for i in df.iloc[0].tolist():

        #elif type(i) in [int, float, np.int64, np.float32]:
        if is_number(i): #if able to cast to numeric values, then use it as a numeric column
            numeric.append(t)
        else: #if not, then use word embeddings. later everything will be cast to string values
            embeddings.append(t)
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
                           phone_number=None,
                           address_columns=[],
                           zip_code=None,
                           geocode_address=False,
                           api_key=None,
                           word_embedding_model="word2vec",
                           word_embedding_path=None,
                           embedding_weight = 'tfidf'):
                           #word_embedding_path='data/embeddings/cc.en.300.bin'):

        """
        This function divides the given raw data into three preprocessed sub-dataset (or numpy matrices):
        - numerical matrix
        - special treatment columns
        - word embedding matrix; shape: (# of attributes, # of entities, dim of word embedding(e.g. 300))
        :param df1: pd.df
        :param df2: pd.df
        :param special_columns: a list of special columns values
        :param phone_number: a list of the name of the phone number column
        :param address_columns: a list of the names of the address columns
        :param geocode_address: bol | indicates whether geocoding should be applied
        :param api_key: str | if geocode_address is true, you must provide a valid Google API key
        :param word_embedding_path: str | path to the embedding dictionary, if None, use default dataset
        :return: dict | a dictionary of np.arrays with the three values
        """
    
        special_columns += address_columns
        if self.phone_number:
            special_columns += [phone_number]
        if self.zip_code:
            special_columns += [zip_code]
        

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


        #process word embeddings
        if divide_col["word_embedding_cols"] and word_embedding_model != 'none': #process only if both col lists are not empty
        
            if word_embedding_model not in ["word2vec","fasttext","glove"]:
                raise ValueError('Invalid model name.')
            
            elif word_embedding_model == "word2vec" and word_embedding_path == None:
                word_embedding_path = 'data/embeddings/GoogleNews-vectors-negative300.bin'
            
            elif word_embedding_model == "fasttext" and word_embedding_path == None:
                word_embedding_path = 'data/embeddings/cc.en.300.bin'
            
            elif word_embedding_model == "glove" and word_embedding_path == None:
                word_embedding_path = 'data/embeddings/glove.42B.300d_word2vec.txt'
                #has to be in non-binary readable by gensim.models.KeyedVectors.load_word2vec_format
                
            embed = Word_embedding(word_embedding_model,word_embedding_path) #initialization may take a while
            df1_embed = embed.dataframe_to_embedding(df1,divide_col["word_embedding_cols"], weight = embedding_weight)
            df2_embed = embed.dataframe_to_embedding(df2,divide_col["word_embedding_cols"], weight = embedding_weight)
            
        else:
            df1_embed = np.array([])
            df2_embed = np.array([])
 
        # process special columns
        if divide_col['special_field_cols']:

            df1_special, lat1,long1 = preprocess_special_fields(df1.iloc[:,
                                                    divide_col['special_field_cols']],
                                                    phone_number,
                                                    address_columns,
                                                    zip_code,
                                                    geocode_address,
                                                    api_key)
            df2_special, lat2,long2 = preprocess_special_fields(df2.iloc[:,
                                                    divide_col['special_field_cols']],
                                                    phone_number,
                                                    address_columns,
                                                    zip_code,
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
            df1_numeric = df1.iloc[:, divide_col['numerical_cols']].as_matrix().astype(float) #some may still be in string type
            df2_numeric = df2.iloc[:, divide_col['numerical_cols']].as_matrix().astype(float) #some may still be in string type

        else:
            df1_numeric = np.array([])
            df2_numeric = np.array([])

        ## after finishing preprocessing
        processed_data = {"numerical":[df1_numeric, df2_numeric],
                          "special_fields":[df1_special, df2_special],
                          "word_embedding_fields":[df1_embed, df2_embed]
                          }
        return processed_data


def load_word_embedding_model(word_embedding_model="word2vec",word_embedding_path=None):

    if word_embedding_model not in ["word2vec","fasttext","glove"]:
        raise ValueError('Invalid model name.')
    
    elif word_embedding_model == "word2vec" and word_embedding_path == None:
        word_embedding_path = 'data/embeddings/GoogleNews-vectors-negative300.bin'
    
    elif word_embedding_model == "fasttext" and word_embedding_path == None:
        word_embedding_path = 'data/embeddings/cc.en.300.bin'
    
    elif word_embedding_model == "glove" and word_embedding_path == None:
        word_embedding_path = 'data/embeddings/glove.42B.300d_word2vec.txt'
        #has to be in non-binary readable by gensim.models.KeyedVectors.load_word2vec_format
    
    return Word_embedding_new(word_embedding_model,word_embedding_path) #initialization may take a while


class Preprocessor():
    def __init__(self,word_embedding_model_instance=None,
                 special_columns=[],
                 phone_number=None,
                 address_columns=[],
                 zip_code=None,
                 geocode_address=False,
                 api_key=None,
                 embedding_weight='tfidf'):
        
        self.phone_number=phone_number
        self.address_columns=address_columns
        self.zip_code=zip_code
        self.geocode_address=geocode_address
        self.api_key=api_key
        self.special_columns = special_columns
        self.special_columns += self.address_columns
        if self.phone_number:
            self.special_columns += [self.phone_number]
        if self.zip_code:
            self.special_columns += [self.zip_code]

        self.divide_col = {"numerical_cols": [],
                      "special_field_cols":[],
                      "word_embedding_cols":[]}
        
        
        self.word_embedding_model_instance=word_embedding_model_instance
        self.word_embed_fit_d1 = None
        self.word_embed_fit_d2 = None
        self.embedding_weight=embedding_weight
        

    def fit(self,df1_to_fit=[],df2_to_fit=[]):

        n, s, w = divide_columns(df1_to_fit, self.special_columns)
        self.divide_col['numerical_cols'] = n
        self.divide_col['special_field_cols'] = s
        self.divide_col['word_embedding_cols'] = w

        print('**** df1 divide columns ****')
        [print(i, ': ', df1_to_fit.columns[j].values) for i, j in self.divide_col.items()]

        print('\n','**** df2 divide columns ****')
        [print(i, ': ', df2_to_fit.columns[j].values) for i, j in self.divide_col.items()]
        
        if len(df1_to_fit)==0 or len(df2_to_fit)==0:
            raise ValueError('Please provide training data for the tf-idf model.')

        
        #fit word embeddings
        if self.divide_col["word_embedding_cols"]: #process only if both col lists are not empty
        
            if self.word_embedding_model_instance==None: #if instance is not provided, use default
                word_embed = load_word_embedding_model()
            else:
                word_embed = self.word_embedding_model_instance

            self.word_embed_fit_d1 = word_embed.fit_embedding(df1_to_fit,self.divide_col["word_embedding_cols"], weight = self.embedding_weight)
            self.word_embed_fit_d2 = word_embed.fit_embedding(df2_to_fit,self.divide_col["word_embedding_cols"], weight = self.embedding_weight)
        
        return self

    def transform(self,df1=[],df2=[]): #df1 and df2 are both df with only one row

        if len(df1)==0 or len(df2)==0:
            raise ValueError('Got empty data.')
        
        if self.word_embed_fit_d1== None or self.word_embed_fit_d2== None:
            raise ValueError('Please fit the preprocessor first.')
        
        #process word embeddings
        if self.divide_col["word_embedding_cols"]: #process only if both col lists are not empty
            
            df1_embed = df_to_embedding(self.word_embed_fit_d1,df1)
            df2_embed = df_to_embedding(self.word_embed_fit_d2,df2)
            
        else:
            df1_embed = np.array([])
            df2_embed = np.array([])
 
        # process special columns
        if self.divide_col['special_field_cols']:

            df1_special, lat1,long1 = preprocess_special_fields(df1.iloc[:,
                                                    self.divide_col['special_field_cols']],
                                                    self.phone_number,
                                                    self.address_columns,
                                                    self.zip_code,
                                                    self.geocode_address,
                                                    self.api_key)
            df2_special, lat2,long2 = preprocess_special_fields(df2.iloc[:,
                                                    self.divide_col['special_field_cols']],
                                                    self.phone_number,
                                                    self.address_columns,
                                                    self.zip_code,
                                                    self.geocode_address,
                                                    self.api_key)

            if self.geocode_address and self.api_key:
                df1['lat'] = lat1
                df1['long'] = long1
                df2['lat'] = lat2
                df2['long'] = long2
                self.divide_col['numerical_cols'] = self.divide_col['numerical_cols'] +\
                                               [-2,-1]

        else:
            df1_special = np.array([])
            df2_special = np.array([])

        # process numeric columns
        if self.divide_col['numerical_cols']:
            df1_numeric = df1.iloc[:, self.divide_col['numerical_cols']].as_matrix().astype(float) #some may still be in string type
            df2_numeric = df2.iloc[:, self.divide_col['numerical_cols']].as_matrix().astype(float) #some may still be in string type

        else:
            df1_numeric = np.array([])
            df2_numeric = np.array([])

        ## after finishing preprocessing
        processed_data = {"numerical":[df1_numeric, df2_numeric],
                          "special_fields":[df1_special, df2_special],
                          "word_embedding_fields":[df1_embed, df2_embed]
                          }
        return processed_data




class Preprocessing_row():
    def __init__(self,df1_to_fit=[],df2_to_fit=[],
                 special_columns=[],
                 phone_number=None,
                 address_columns=[],
                 zip_code=None,
                 geocode_address=False,
                 api_key=None,
                 word_embedding_model="word2vec",
                 word_embedding_path=None,
                 embedding_weight = 'tfidf'):
        
        if len(df1_to_fit)==0 or len(df2_to_fit)==0:
            raise ValueError('Please provide training data for the tf-idf model.')
        
        self.phone_number=phone_number
        self.address_columns=address_columns
        self.zip_code=zip_code
        self.geocode_address=geocode_address
        self.api_key=api_key
        self.special_columns = special_columns
        self.special_columns += self.address_columns
        if self.phone_number:
            self.special_columns += [self.phone_number]
        if self.zip_code:
            self.special_columns += [self.zip_code]

        self.divide_col = {"numerical_cols": [],
                      "special_field_cols":[],
                      "word_embedding_cols":[]}

        n, s, w = divide_columns(df1_to_fit, special_columns)
        self.divide_col['numerical_cols'] = n
        self.divide_col['special_field_cols'] = s
        self.divide_col['word_embedding_cols'] = w

        print('**** df1 divide columns ****')
        [print(i, ': ', df1_to_fit.columns[j].values) for i, j in self.divide_col.items()]

        print('\n','**** df2 divide columns ****')
        [print(i, ': ', df2_to_fit.columns[j].values) for i, j in self.divide_col.items()]
        
        
        self.word_embed_fit_d1 = None
        self.word_embed_fit_d2 = None
        
        #fit word embeddings
        if self.divide_col["word_embedding_cols"]: #process only if both col lists are not empty
        
            if word_embedding_model not in ["word2vec","fasttext","glove"]:
                raise ValueError('Invalid model name.')
            
            elif word_embedding_model == "word2vec" and word_embedding_path == None:
                word_embedding_path = 'data/embeddings/GoogleNews-vectors-negative300.bin'
            
            elif word_embedding_model == "fasttext" and word_embedding_path == None:
                word_embedding_path = 'data/embeddings/cc.en.300.bin'
            
            elif word_embedding_model == "glove" and word_embedding_path == None:
                word_embedding_path = 'data/embeddings/glove.42B.300d_word2vec.txt'
                #has to be in non-binary readable by gensim.models.KeyedVectors.load_word2vec_format
            
            word_embed = Word_embedding_row(word_embedding_model,word_embedding_path) #initialization may take a while
            self.word_embed_fit_d1 = word_embed.fit_embedding(df1_to_fit,self.divide_col["word_embedding_cols"], weight = embedding_weight)
            self.word_embed_fit_d2 = word_embed.fit_embedding(df2_to_fit,self.divide_col["word_embedding_cols"], weight = embedding_weight)
            

    def process_zipcode(self):
        pass
    def process_phone_num(self):
        pass

    def overall_preprocess(self,df1=[],df2=[]): #df1 and df2 are both df with only one row


        if len(df1)==0 or len(df2)==0:
            raise ValueError('Got empty data.')
        
        if self.word_embed_fit_d1== None or self.word_embed_fit_d2== None:
            raise ValueError('Please fit the preprocessor first.')
        
        #process word embeddings
        if self.divide_col["word_embedding_cols"]: #process only if both col lists are not empty
            
            df1_embed = row_to_embedding(self.word_embed_fit_d1,df1)
            df2_embed = row_to_embedding(self.word_embed_fit_d2,df2)
            
        else:
            df1_embed = np.array([])
            df2_embed = np.array([])
 
        # process special columns
        if self.divide_col['special_field_cols']:

            df1_special, lat1,long1 = preprocess_special_fields(df1.iloc[:,
                                                    self.divide_col['special_field_cols']],
                                                    self.phone_number,
                                                    self.address_columns,
                                                    self.zip_code,
                                                    self.geocode_address,
                                                    self.api_key)
            df2_special, lat2,long2 = preprocess_special_fields(df2.iloc[:,
                                                    self.divide_col['special_field_cols']],
                                                    self.phone_number,
                                                    self.address_columns,
                                                    self.zip_code,
                                                    self.geocode_address,
                                                    self.api_key)

            if self.geocode_address and self.api_key:
                df1['lat'] = lat1
                df1['long'] = long1
                df2['lat'] = lat2
                df2['long'] = long2
                self.divide_col['numerical_cols'] = self.divide_col['numerical_cols'] +\
                                               [-2,-1]

        else:
            df1_special = np.array([])
            df2_special = np.array([])

        # process numeric columns
        if self.divide_col['numerical_cols']:
            df1_numeric = df1.iloc[:, self.divide_col['numerical_cols']].as_matrix().astype(float) #some may still be in string type
            df2_numeric = df2.iloc[:, self.divide_col['numerical_cols']].as_matrix().astype(float) #some may still be in string type

        else:
            df1_numeric = np.array([])
            df2_numeric = np.array([])

        ## after finishing preprocessing
        processed_data = {"numerical":[df1_numeric, df2_numeric],
                          "special_fields":[df1_special, df2_special],
                          "word_embedding_fields":[df1_embed, df2_embed]
                          }
        return processed_data
