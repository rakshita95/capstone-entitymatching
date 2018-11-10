
# coding: utf-8

# In[4]:

import sys
import pytest
import numpy as np
import pandas as pd

sys.path.append('..')
from modules.preprocessing.generate_labels import gen_labels

def test_gen_labels():
    df1_id = pd.Series([1,2])
    df2_id = pd.Series(['a','b'])
    match = pd.DataFrame(['a','b'],[1,2]).reset_index()
    match.columns=(['id1','id2'])
    
    tmp = gen_labels(df1_id, df2_id, match)
    desired = np.array([1,0,0,1])
    
    assert(np.array_equal(tmp,desired))


# In[ ]:



