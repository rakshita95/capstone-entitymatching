import numpy as np
import sys
#sys.path.append('..')
from preprocessing import word_embeddings

class Preprocessing():
    def _init__(self):
        pass
    def process_zipcode(self):
        pass
    def process_phone_num(self):
        pass
    def process_text(self):
        pass
    def word_embeddings(self):
        """
        This function calls all preprocessing functions for texts
        then it takes those fields to generate word embeddings
        :return:
        """
        pass

    def overall_preprocess(self):
        """
        This function divide the given raw data into three sub-dataset (or numpy matrices):
        - numerical matrix
        - word embedding matrix
        - special treatment columns


        e.g. word embedding matrix should have shape:
        (number of samples, number of fields,embedding size (e.g. 300))

        :return: three matrices
        """
        pass



