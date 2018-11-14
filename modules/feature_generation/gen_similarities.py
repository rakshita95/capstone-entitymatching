'''
Serena Zhang
Nov 5 2018
'''

import itertools

import numpy as np
from jellyfish import levenshtein_distance
import scipy.spatial as sp

class similarities():

    def __init__(self):
        pass

    @staticmethod
    def __gen_cross_product(matrix_1,matrix_2,func,embedding = False):
        """
        This function gets called when a certain type of similarity is calculated.
        It is a private function that can only be called within similarities()

        :param matrix_1, matrix2:
        two numpy matrices that share the same shape[1] (i.e. number of fields(price, year etc)),
        but with different shape[0] (i.e number of samples)
        there might or might not be shape[2] as it can be a vector or a value
        :param func: function to calculate similarity
        :param num: whether it calculates numerical similarities. default is True
        :return:
        """
        matrix_1_num_sample, matrix_2_num_sample = matrix_1.shape[0], matrix_2.shape[0]
        num_fields = matrix_1.shape[1] # same as matrix_2.shape[2]

        output = np.empty(shape = (matrix_1_num_sample*matrix_2_num_sample,num_fields))
        for (ind, (array_1,array_2)) in enumerate(itertools.product(matrix_1,matrix_2)):
            if embedding == True:
                output[ind] = func(array_1,array_2)
            else:
                output[ind] = np.apply_along_axis(func,0,array_1,array_2)
        return output

    def numerical_similarity_on_matrix(self,matrix_1, matrix_2,method = "scaled_gaussian"):
        """
        calculates numerical similarities for all numerical fields

        :param matrix_1: numpy matrix of numerical fields for dataset a
        :param matrix_2: numpy matrix of a numerical field for dataset b.
        both matrices should have the same shape[1] - number of fields
        with potentially different shape[0] - number of samples
        Order of fields should also be aligned
        :param method: method to calculate scaler similarity
        :return: a 2-D array of similarities for all numerical records
        of allpossible combination of samples
        """
        if not(matrix_1) or not(matrix_2):
            print("empty matrix - could not calculate similarity")
            return np.array([])

        def scaled_gaussian(a,b):
            return np.exp(-2 * abs(a - b) / (a + b + 1e-5))
        def min_max(a,b):
            return np.round(np.min((a, b), axis=0) / np.max((a, b), axis=0), 3)

        if method == "scaled_gaussian":
            out = similarities().__gen_cross_product(matrix_1, matrix_2, scaled_gaussian)

        elif method == "min_max":
            out = similarities().__gen_cross_product(matrix_1, matrix_2, min_max)
        return out

    def vector_similarity_on_matrix(self, matrix_1, matrix_2, method = "cosine"):
        '''
        calculates text similarities given word embeddings fields

        :param matrix_1, matrix_2:
        both matrices should have the same shape[1] - number of fields
        both matrices should have the same shape[2] - embedding size
        with potentially different shape[0] - number of samples
        Order of fields should also be aligned
        :return:
        '''
        if matrix_1.size != 0 and matrix_2.size != 0:  # if both are not empty matrices
            n_cols = matrix_1.shape[1]
            out = np.zeros(
                (matrix_1.shape[0] * matrix_2.shape[0], n_cols))
            for col in range(n_cols):
                if method == 'cosine':
                    sim = 1 - sp.distance.cdist(matrix_1[:, col, :],
                                                matrix_2[:, col, :],
                                                'cosine')
                out[:, col] = sim.flatten('C')
            return out

        else:  # else return empty array
            return np.array([])


    def text_similarity_on_matrix(self,matrix_1,matrix_2, method = "lavenshtein"):
        """

        :param matrix_1,matrix_2:
        both matrices should have the same shape[1] - number of fields
        with potentially different shape[0] - number of samples
        Order of fields should also be aligned
        :param method: specified similarity metric to use
        :return: a 2-D array of similarities for all special text records
        of all possible combination of samples
        """
        if not(matrix_1) or not(matrix_2):
            print("empty matrix - could not calculate similarity")
            return np.array([])
        def lavenshtein(a,b):
            tmp = [levenshtein_distance(x, y) for i, x in enumerate(a) for j, y in enumerate(b) if i == j]
            return np.asarray(tmp)
        if method == "lavenshtein":
            out = similarities().__gen_cross_product(matrix_1,matrix_2,lavenshtein)
        return out

    def generate_similarity(self, matrix1, matrix2):
        """
        This function takes in the preprocessed matrices and calcualtes the
        similarity between the different entries - it is assumed that the
        columns are in the same order.

        :param matrix1: matrix with the preprocessed features
        :param matrix2: matrix that is supposed to be matches
        :return: feature matrix for the machine learning model
        """
        if not(matrix1) or not(matrix2):
            print("empty matrix - could not calculate similarity")
            return np.array([])

        def get_indices(matrix):
            t = 0
            embeddings = []
            special = []
            numeric = []
            for i in matrix[0]:
                if type(i) == list:
                    embeddings.append(t)
                elif type(i) == float or type(i) == int:
                    numeric.append(t)
                elif type(i) == str:
                    special.append(t)
                t += 1
            return embeddings, special, numeric

        # split matrix

        embeddings, special, numeric = get_indices(matrix1)

        # apply the functions

        embeddings_sim = similarities().vector_similarity_on_matrix(matrix1[:,embeddings],
                                                     matrix2[:,embeddings])
        special_sim = similarities().text_similarity_on_matrix(matrix1[:,special],
                                                     matrix2[:,special])
        numeric_sim = similarities().numerical_similarity_on_matrix(matrix1[:,numeric].astype(float),
                                                     matrix2[:,numeric].astype(float))

        # concatenate it and return

        features = np.concatenate((embeddings_sim,
                                   special_sim,
                                   numeric_sim),
                                  axis=1)

        return features

if __name__ == "__main__":
    pass

