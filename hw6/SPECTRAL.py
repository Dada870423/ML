import numpy as np

from KMEANS import *


def Spectral(Gram, k):
    # degree matrix
    Degree = np.diag(np.sum(Gram, axis=1))
    Laplacian = Degree - Gram
    D__1_2 = np.diag(1 / np.sqrt(Degree[Degree!=0]))
    Laplacian_sym = D__1_2 @ Laplacian @ D__1_2

    ## compute the first k eigenvectors
    eigen_values, eigen_vectors = np.linalg.eig(Laplacian_sym)
    saved_vectors = "./vector_S_8_C_5.npy"
    saved_value = "./value_S_8_C_5.npy"
    np.save(saved_vectors, eigen_vectors)
    np.save(saved_values, eigen_values)
    print(eigen_values)

