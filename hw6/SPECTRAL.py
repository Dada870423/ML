import numpy as np
import os

from KMEANS import *


def Spectral(Gram, k):
    # degree matrix
    Degree = np.diag(np.sum(Gram, axis=1))
    Laplacian = Degree - Gram
    D__1_2 = np.diag(1 / np.sqrt(Degree[Degree!=0]))
    Laplacian_sym = D__1_2 @ Laplacian @ D__1_2

    ## compute the first k eigenvectors
    saved_vectors = "./vector_S_8_C_5.npy"
    saved_values = "./value_S_8_C_5.npy"

    saved_vectors_sort = "./vector_S_8_C_5_sort.npy"
    saved_values_sort = "./value_S_8_C_5_sort.npy"
    if os.path.isfile(saved_vectors):
        eigen_values_unsorted, eigen_vectors_unsorted = np.linalg.eig(Laplacian_sym)
        np.save(saved_vectors, eigen_vectors)
        np.save(saved_values, saved_value)
    else:
        eigen_vectors_unsorted = np.load("vector_S_8_C_5.npy")
        eigen_values_unsorted = np.load("value_S_8_C_5.npy")
        Eigen_index = np.argsort(eigen_values_unsorted)
        Eigen_Value = eigen_values_unsorted[Eigen_index]
        Eigen_Vector = eigen_vectors_unsorted[Eigen_index]
        np.save(saved_vectors_sort, saved_vectors_sort)
        np.save(saved_values_sort, Eigen_Value)



