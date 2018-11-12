import sys
import pytest
import numpy as np
import pandas as pd

sys.path.append('..')
from modules.preprocessing.generate_labels import gen_labels

def test_gen_labels():
    df1_id = pd.Series([1,2])
    df2_id = pd.Series(['a','b'])
    match = pd.DataFrame([1,2],['a','b']).reset_index()
    match.columns=(['id2','id1'])
    
    tmp = gen_labels(df1_id, df2_id, match, 'id1','id2')
    desired = np.array([1,0,0,1])

    assert(np.array_equal(tmp,desired))