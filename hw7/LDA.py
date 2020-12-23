from UTIL import *

import numpy as np
from scipy import linalg

def LDA(images, Size, label, FacePath):
    cate = np.unique(np.array(label)[:, 1])
    index = np.array([np.where(np.array(label)[:, 1] == iter_cate) for iter_cate in cate], dtype = object)

    mean = np.mean(images, axis = 0)

    print("---------", mean.shape)




    for iter_cate in range(len(cate)):
        print(index[iter_cate][0])


    cate_mean = np.zeros((len(cate), Size[0] * Size[1]))
    print(cate_mean.shape)

    for iter_mean in range(len(cate_mean)):
        cate_mean[iter_mean] = np.mean(images[index[iter_mean][0]], axis = 0)



    S_W = np.zeros((Size[0] * Size[1], Size[0] * Size[1]))

    S_B = np.zeros((Size[0] * Size[1], Size[0] * Size[1]))

    print(images[index[iter_cate][0]].shape[0])

    #W = (feature.T @ images.T).T ## maybe 135 * 135
    #W_25 = (feature.T @ feature.T[:25].T).T
    #print("W: ", W.shape)
    #print("W_25", W_25.shape)




    for iter_cate in range(len(cate)):
        y_m = images[index[iter_cate][0]] - cate_mean[iter_cate]
        m_m = cate_mean[iter_cate] - mean[iter_cate]

        print("y_m", y_m.shape)
        print("m_m", m_m.shape)


        S_W += y_m.T @ y_m
        S_B += images[index[iter_cate][0]].shape[0] * m_m.T @ m_m
        print("(m_m.T @ m_m).shape", (m_m.T @ m_m).shape)

    print("####", (linalg.inv(S_W) @ S_B).shape)



    eigen_values_unsorted, eigen_vectors_unsorted = linalg.eig(linalg.inv(S_W) @ S_B)

    #print("linalg.inv(S_W) @ S_B", (linalg.inv(S_W) @ S_B).shape)

    print("eigen_vectors_unsorted", eigen_vectors_unsorted.shape)
    print("eigen_values_unsorted", eigen_values_unsorted.shape)


    #eigen_vec = Sort_Norm_eigen(eigen_values_unsorted, eigen_vectors_unsorted, mean, images)

    Eigen_index = np.argsort(eigen_values_unsorted)
    Eigen_Value = eigen_values_unsorted[Eigen_index]
    Eigen_Vector = eigen_vectors_unsorted[Eigen_index].real

    print("mean:", mean.shape)
    print("Eigen_Vector", Eigen_Vector.shape)



    eigen_vec = (Eigen_Vector[0:135].T @ images).T

    






    print("eigen_vec", eigen_vec.shape)

    print("images", images.shape)

    for iter_vector in range(25):
        eigen_vec[:, iter_vector] /= np.linalg.norm(eigen_vec[:, iter_vector])

    EigenFace(Eigenface = eigen_vec, path = FacePath, Size = Size)
    return cate_mean, eigen_vec






