from UTIL import *

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist, squareform

def PCA(images, Size, FacePath):
    mean = np.mean(images, axis = 0)

    covariance = (images - mean) @ (images - mean).T


    covariance /= len(images)


    print("covariance", covariance.shape)
    print(covariance)
    eigen_values_unsorted, eigen_vectors_unsorted = linalg.eig(covariance)


#    eigen_vec = Sort_Norm_eigen(eigen_values_unsorted, eigen_vectors_unsorted, mean, images)
    Eigen_index = -np.argsort(eigen_values_unsorted)
    Eigen_Value = eigen_values_unsorted[Eigen_index]
    Eigen_Vector = eigen_vectors_unsorted[Eigen_index].real


    eigen_face = ((images - mean).T @ Eigen_Vector[:25].T) ## 2500 * 25

    W = ((images - mean) @ (images - mean).T @ Eigen_Vector[:25].T) ## 135 * 25






    if FacePath != None:
        EigenFace(Eigenface = eigen_face.T, path = FacePath, Size = Size)
    return mean, eigen_face, W


def PCA_KNN(k, images, EigenFace, proj_train_image, label, test_images, test_label):
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
    print("outcome:", it)



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



    eigen_values_unsorted, eigen_vectors_unsorted = linalg.eig(Kernel)

    Eigen_index = -np.argsort(eigen_values_unsorted)
    Eigen_Value = eigen_values_unsorted[Eigen_index]
    Eigen_Vector = eigen_vectors_unsorted[Eigen_index].real

    mean = np.mean(images, axis = 0)


    eigen_face = ((images - mean).T @ Eigen_Vector[:25].T) ## 2500 * 25

    W = ((images - mean) @ (images - mean).T @ Eigen_Vector[:25].T) ## 135 * 25

    return mean, eigen_face, W






