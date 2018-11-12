import pandas as pd
import numpy as np
import itertools

def gen_labels(df1_id, df2_id, match_df, match_id1, match_id2):
    """
    Generate y labels for the final cross product data
    
    *** PAY CLOSE ATTENTION TO THE ORDER OF THE INPUTS ***
    
    :param df1_id: the id column of the first data frame
    :param df2_id: the id column of the second data frame
    **Order of data frames should be aligned with the similarity functions**
    
    :param match_df: a data frame containing the pairs of ids that are matches 
    :param match_id1: column name in match_df corresponding to df1_id
    :param match_id2: column name in match_df corresponding to df2_id
    
    :return y: Labels of the cross product
    
    """ 

    cross = []
    # Concatenate each matched id pair into a string
    match = match_df[match_id1].astype(str) +'_'+ match_df[match_id2].astype(str)
    
    # Get all possible combinations of id pairs
    # Concatenate each pair into a string
    for (id_1,id_2) in itertools.product(list(df1_id), list(df2_id)):
        cross.append(str(id_1) +'_'+ str(id_2))
    
    # Generate labels
    labels = np.array(pd.Series(cross).isin(match).astype(int))
    
    return labels