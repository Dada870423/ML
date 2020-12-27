from UTIL import *

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist, squareform


def PCA(images, Size, FacePath):
    ## mean
    mean = np.mean(images, axis = 0)

    ## covariance
    covariance = ((images - mean) @ (images - mean).T) / len(images)

    ## sort the eigenvalue, and get the 25 largest eigenvectors
    eigen_values_unsorted, eigen_vectors_unsorted = np.linalg.eig(covariance)
    Eigen_index = -np.argsort(eigen_values_unsorted)
    Eigen_Value = eigen_values_unsorted[Eigen_index]
    Eigen_Vector = eigen_vectors_unsorted[Eigen_index].real

    ## project images on eigenvectors
    eigen_face = ((images - mean).T @ Eigen_Vector[:25].T) ## 2500 * 25
    W = ((images - mean) @ (images - mean).T @ Eigen_Vector[:25].T) ## 135 * 25

    if FacePath != None:
        ## plot eigenface
        EigenFace(Eigenface = eigen_face.T, path = FacePath, Size = Size)
    return mean, eigen_face, W



## ref(https://arbu00.blogspot.com/2017/02/7-kernel-pca.html)
def PCA_Kernel(images, Gamma, method = 0):
    if method == 0: ## linear
        print("linear kernel")
        pre_Kernal = images @ images.T
    else: ## RBF
        print("RBF kernel")
        pre_Kernal = squareform(np.exp(- Gamma * pdist(images, "sqeuclidean")))

    N1 = np.ones((len(images), len(images))) / len(images)
    Kernel = pre_Kernal - N1 @ pre_Kernal - N1 @ pre_Kernal + N1 @ pre_Kernal @ N1



    eigen_values_unsorted, eigen_vectors_unsorted = np.linalg.eig(Kernel)

    Eigen_index = -np.argsort(eigen_values_unsorted)
    Eigen_Value = eigen_values_unsorted[Eigen_index]
    Eigen_Vector = eigen_vectors_unsorted[Eigen_index].real

    mean = np.mean(images, axis = 0)

    ## compute the eigen face, first 25 eigenvectors
    eigen_face = ((images - mean).T @ Eigen_Vector[:25].T) ## 2500 * 25

    ## Dimension reduction, which is images project on eigenvectors
    W = ((images - mean) @ (images - mean).T @ Eigen_Vector[:25].T) ## 135 * 25

    return mean, eigen_face, W



def PCA_1(images, Size, FacePath):
    mean = np.tile(np.mean(images, axis = 0), (len(images), 1))

    covariance = (images - mean).T @ (images - mean)


    covariance /= len(images)



    eigen_values_unsorted, eigen_vectors_unsorted = np.linalg.eig(covariance)

    Eigen_index = -np.argsort(eigen_values_unsorted)
    Eigen_Value = eigen_values_unsorted[Eigen_index]
    Eigen_Vector = eigen_vectors_unsorted[Eigen_index].real
    Eigen_Vector = eigen_vectors_unsorted.real

    eigen_face = Eigen_Vector[:, :25]
    W = images @ Eigen_Vector[:, 0:25]

    if FacePath != None:
        EigenFace(Eigenface = eigen_face.T, path = FacePath, Size = Size)
    return mean, eigen_face, W




