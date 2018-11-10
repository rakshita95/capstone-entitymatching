
# coding: utf-8

# In[214]:

import pandas as pd
import numpy as np
import itertools

def gen_labels(df1_id, df2_id, match_df):
    """
    Generate labels of the 
    
    *** PLEASE READ CAREFULLY THE DESCRIPTION BELOW    ***
    *** PAY CLOSE ATTENTION TO THE ORDER OF THE INPUTS ***
    
    :param df1_id: the id column of the first data frame
    :param df2_id: the id column of the second data frame
    **Order of data frames should be aligned with the similarity functions**
    :param match_df: a data frame containing the pairs of ids that are matches 
                     The first column should correspond to ids from df1
                     Second column should correspond to ids from df2
    :return y: Labels of the cross product
    
    """ 

    cross = []
    # Concatenate each matched id pair into a string
    match = list(match_df.iloc[:,0].astype('str') +'_'+ match_df.iloc[:,1].astype('str'))
    
    # Get all possible combinations of id pairs
    # Concatenate each pair into a string
    for (id_1,id_2) in itertools.product(list(df1_id), list(df2_id)):
        cross.append(str(id_1) +'_'+ str(id_2))
    
    # Combine matched pairs with all other pairs, and generate labels
    labels = np.array([item in match for item in cross]).astype('int')
    
    return labels


# In[ ]:



