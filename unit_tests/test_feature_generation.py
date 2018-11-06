import os
import sys
import pytest

import numpy as np

sys.path.append('..')
from modules.feature_generation.gen_similarities import similarities


def test_numerical_similarity_on_matrix():
    matrix_1 = np.array([[1,2,3,4],[5,6,7,8]])
    matrix_2 = np.array([[1,2,3,4]])

    tmp = similarities().numerical_similarity_on_matrix(matrix_1,matrix_2,method = "min_max")
    assert(np.array_equal(tmp,np.array([[1,1,1,1],[0.2 ,0.333, 0.429, 0.5 ]])))

def test_vector_similarity():
    matrix_1 = np.array([[[1,1,1], [1,2,3],[1,3,1]],
                         [[3,6,7],[2,3,1],[1,1,1]]])
    matrix_2 = np.array([[[2,2,2],[2,3,4],[1,1,1]]])
    tmp = similarities().vector_similarity(matrix_1,matrix_2)
    desired = np.array(  [[1, 0.993,0.87],[0.953, 0.844, 1 ]])
    assert(np.array_equal(tmp,desired))