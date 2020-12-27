from UTIL import *

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist, squareform

def LDA(images, Size, label, FacePath):
    ## get the all category
    cate = np.unique(np.array(label)[:, 1])
    index = np.array([np.where(np.array(label)[:, 1] == iter_cate) for iter_cate in cate], dtype = object)

    mean = np.mean(images, axis = 0)

    ## calculate mean of each category
    cate_mean = np.zeros((len(cate), Size[0] * Size[1]))

    for iter_mean in range(len(cate_mean)):
        cate_mean[iter_mean] = np.mean(images[index[iter_mean][0]], axis = 0)

    ## S_W is within class, S_B is between class
    S_W = np.zeros((Size[0] * Size[1], Size[0] * Size[1]))
    S_B = np.zeros((Size[0] * Size[1], Size[0] * Size[1]))


    ## calculate based on formuula
    for iter_cate in range(len(cate)):
        y_m = images[index[iter_cate][0]] - cate_mean[iter_cate]
        m_m = cate_mean[iter_cate] - mean[iter_cate]

        S_W += y_m.T @ y_m
        S_B += images[index[iter_cate][0]].shape[0] * m_m.T @ m_m

    eigen_values_unsorted, eigen_vectors_unsorted = np.linalg.eig(linalg.inv(S_W) @ S_B)


    ## sort the eigenvalue, and get the 25 largest eigenvectors
    Eigen_index = np.argsort(eigen_values_unsorted)
    Eigen_Value = eigen_values_unsorted[Eigen_index]
    Eigen_Vector = eigen_vectors_unsorted[Eigen_index].real

    ## project images on eigenvectors
    eigen_face = (Eigen_Vector[0:135].T @ images).T
    W = images @ (Eigen_Vector[:, 0:25])

    if FacePath != None:
        ## plot eigenface
        EigenFace(Eigenface = eigen_face.T, path = FacePath, Size = Size)

    return cate_mean, eigen_face[:25].T, W


def LDA_Kernel(images, Gamma, label, method = 0):
    if method == 0: ## linear kernel
        print("linear kernel")
        pre_Kernal = images @ images.T
    else: ## RBF kernel
        print("RBF kernel")
        pre_Kernal = squareform(np.exp(- Gamma * pdist(images, "sqeuclidean")))
    ## get the all category
    cate = np.unique(np.array(label)[:, 1])
    index = np.array([np.where(np.array(label)[:, 1] == iter_cate) for iter_cate in cate], dtype = object)

    mean = np.mean(pre_Kernal, axis = 0)

    ## calculate mean of each category
    cate_mean = np.zeros((len(cate), len(pre_Kernal)))
    for iter_mean in range(len(cate_mean)):
        cate_mean[iter_mean] = np.mean(pre_Kernal[index[iter_mean][0]], axis = 0)

    ## S_W is within class, S_B is between class
    S_W = np.zeros((len(pre_Kernal), len(pre_Kernal)))
    S_B = np.zeros((len(pre_Kernal), len(pre_Kernal)))

    for iter_cate in range(len(cate)):
        y_m = pre_Kernal[index[iter_cate][0]] - cate_mean[iter_cate]
        m_m = cate_mean[iter_cate] - mean[iter_cate]
        S_W += y_m.T @ y_m
        S_B += pre_Kernal[index[iter_cate][0]].shape[0] * m_m.T @ m_m

    ## sort the eigenvalue, and get the 25 largest eigenvectors
    eigen_values_unsorted, eigen_vectors_unsorted = np.linalg.eig(linalg.pinv(S_W) @ S_B)

    Eigen_index = -np.argsort(eigen_values_unsorted)
    Eigen_Value = eigen_values_unsorted[Eigen_index]
    Eigen_Vector = eigen_vectors_unsorted[Eigen_index].real

    ## project images on eigenvectors
    eigen_face = ((images).T @ Eigen_Vector[:25].T) ## 2500 * 25
    W = ((images) @ (images).T @ Eigen_Vector[:25].T) ## 135 * 25

    return cate_mean, eigen_face, W
