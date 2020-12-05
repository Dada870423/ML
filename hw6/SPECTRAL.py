import numpy as np
import os

from KMEANS import *
from scipy import linalg


def Spectral(Gram, k):
    

    ## compute the first k eigenvectors

    saved_vectors_sort = "./vector_S_75_C_52_sort.npy"
    saved_values_sort = "./value_S_75_C_52_sort.npy"
    if not os.path.isfile(saved_vectors_sort):
        # degree matrix
        Degree = np.diag(np.sum(Gram, axis=1))
        Laplacian = Degree - Gram
        D__1_2 = np.diag(1 / np.sqrt(Degree[Degree!=0]))
        Laplacian_sym = D__1_2 @ Laplacian @ D__1_2

        print("no file")
        #eigen_values_unsorted, eigen_vectors_unsorted = np.linalg.eig(Laplacian_sym)
        Eigen = linalg.eig(Laplacian_sym)
        eigen_values_unsorted = Eigen[0]
        eigen_vectors_unsorted = Eigen[1]


        Eigen_index = np.argsort(eigen_values_unsorted)
        Eigen_Value = eigen_values_unsorted[Eigen_index]
        Eigen_Vector = eigen_vectors_unsorted[Eigen_index]
        np.save(saved_vectors_sort, Eigen_Vector)
        np.save(saved_values_sort, Eigen_Value)
    else:
        Eigen_Vector = np.load(saved_vectors_sort)
        Eigen_Value = np.load(saved_values_sort)
        print(Eigen_Value)
        print(Eigen_Vector.shape)


    ## normalize
    Eigen_Vector_T = np.zeros((Eigen_Vector.shape))

    Eigen_Vector_trans = Eigen_Vector.T

    for iter_row in range(len(Eigen_Vector_trans)):
        SSum = np.sum(Eigen_Vector_trans[iter_row])
        Eigen_Vector_T[iter_row] = Eigen_Vector_trans[iter_row] / SSum

    GIF = Kmeans(Gram = Eigen_Vector_T, k = k)
    return GIF








