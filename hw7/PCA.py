from READFILE import *

import numpy as np
from scipy import linalg

def PCA(images, Size, FacePath):
    mean = np.mean(images, axis = 0)

    covariance = (images - mean) @ (images - mean).T


    covariance /= len(images)


    print("covariance", covariance.shape)
    print(covariance)
    eigen_values_unsorted, eigen_vectors_unsorted = linalg.eig(covariance)

    Eigen_index = np.argsort(eigen_values_unsorted)
    Eigen_Value = eigen_values_unsorted[Eigen_index]
    Eigen_Vector = eigen_vectors_unsorted[Eigen_index]

    print("mean:", mean.shape)
    print("Eigen_Vector", Eigen_Vector.shape)

    eigen_vec = ((images - mean).T @ Eigen_Vector)[:, :25]



    for iter_vector in range(25):
        eigen_vec[:, iter_vector] /= np.linalg.norm(eigen_vec[:, iter_vector])

    EigenFace(Eigenface = eigen_vec, path = FacePath, Size = Size)
    return mean, eigen_vec





