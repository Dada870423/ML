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
    print("fuck--------", cate_mean[0])



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


    #eigen_face = (Eigen_Vector[0:135].T @ images).T
    eigen_face = (Eigen_Vector[0:135].T @ images).T

    W = images @ (Eigen_Vector[0:25]).T


    if FacePath != None:
        print("in if", eigen_face.shape)
        EigenFace(Eigenface = eigen_face.T, path = FacePath, Size = Size)

    return cate_mean, eigen_face[:25].T, W


def LDA_KNN(k, images, EigenFace, proj_train_image, label, test_images, test_label):
    print("test_images", test_images.shape) ## 30 * 2500
    mean = np.mean(images, axis = 0)
    print("EigenFace", EigenFace.shape)
    #proj_train_image = W ## 135 * 25
    proj_test_image = ((test_images - mean) @ EigenFace.T) ## 30 * 25

    just_label = np.array(label)

    proj_train_image_norm = np.zeros((proj_train_image.shape))
    for iter_norm in range(len(proj_train_image_norm)):
        proj_train_image_norm[iter_norm] = normalization(proj_train_image[iter_norm])



    ## label : 135 * 2
    print("proj_test_image", proj_test_image.shape)
    print("proj_train_image", proj_train_image.shape)
    print("proj_train_image_norm", proj_train_image_norm.shape)
    it = 0
    for iter_test_image in range(len(proj_test_image)):
        distance = np.zeros(len(label))
        #print(proj_train_image_norm[0].shape)
        #print("proj_test_image", proj_test_image[0].shape)
        for iter_dis in range(len(label)):
            distance[iter_dis] = np.linalg.norm(proj_train_image_norm[iter_dis] - proj_test_image[iter_test_image])
        outcome = list(just_label[np.argsort(distance)[:k]][:, 1])
        print(outcome, "ans:  ", test_label[iter_test_image])
        #print(distance[np.argsort(distance)[:k]], "\n\n")

        ans = outcome[np.argmax([outcome.count(outcome[i]) for i in range(len(outcome))])]
        if ans == test_label[iter_test_image][1]:
            print("hit!!!!")
            it += 1
    print("LDA : outcome:", it)











def LDA_Kernel(images, Gamma, label, method = 0):
    if method == 0: ## linear
        print("linear kernel")
        pre_Kernal = images @ images.T
    else: ## RBF
        print("RBF kernel")
        pre_Kernal = squareform(np.exp(- Gamma * pdist(images, "sqeuclidean")))

    cate = np.unique(np.array(label)[:, 1])
    index = np.array([np.where(np.array(label)[:, 1] == iter_cate) for iter_cate in cate], dtype = object)

    mean = np.mean(pre_Kernal, axis = 0)







    cate_mean = np.zeros((len(cate), len(pre_Kernal)))
    print(cate_mean.shape)

    for iter_mean in range(len(cate_mean)):
        cate_mean[iter_mean] = np.mean(pre_Kernal[index[iter_mean][0]], axis = 0)



    S_W = np.zeros((len(pre_Kernal), len(pre_Kernal)))

    S_B = np.zeros((len(pre_Kernal), len(pre_Kernal)))

    for iter_cate in range(len(cate)):
        y_m = pre_Kernal[index[iter_cate][0]] - cate_mean[iter_cate]
        m_m = cate_mean[iter_cate] - mean[iter_cate]

        print("y_m", y_m.shape)
        print("m_m", m_m.shape)


        S_W += y_m.T @ y_m
        S_B += pre_Kernal[index[iter_cate][0]].shape[0] * m_m.T @ m_m
        print("(m_m.T @ m_m).shape", (m_m.T @ m_m).shape)

    eigen_values_unsorted, eigen_vectors_unsorted = linalg.eig(linalg.inv(S_W) @ S_B)

    #print("linalg.inv(S_W) @ S_B", (linalg.inv(S_W) @ S_B).shape)

    print("eigen_vectors_unsorted", eigen_vectors_unsorted.shape)
    print("eigen_values_unsorted", eigen_values_unsorted.shape)


    #eigen_vec = Sort_Norm_eigen(eigen_values_unsorted, eigen_vectors_unsorted, mean, images)

    Eigen_index = np.argsort(eigen_values_unsorted)
    Eigen_Value = eigen_values_unsorted[Eigen_index]
    Eigen_Vector = eigen_vectors_unsorted[Eigen_index].real




    eigen_face = ((images).T @ Eigen_Vector[:25].T) ## 2500 * 25

    W = ((images) @ (images).T @ Eigen_Vector[:25].T) ## 135 * 25

    return cate_mean, eigen_face, W
