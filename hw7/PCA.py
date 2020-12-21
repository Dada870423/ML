import numpy as np
from scipy import linalg

def PCA(images):
    mean = np.mean(images, axis = 0)
    #x_x_bar = images - mean
    #print("x_x_bar", x_x_bar.shape)
    #print("x_x_bar.T", x_x_bar.T.shape)
    #x_x_bar_at_x_x_bar_T = (x_x_bar @ x_x_bar.T) 
    #print("x_x_bar_at_x_x_bar_T", x_x_bar_at_x_x_bar_T.shape)
    #covariance = x_x_bar_at_x_x_bar_T / len(iamges)
    #print("covariance", covariance.shape)
    covariance = np.zeros((50, 50))
    for iter_image in range(len(images)):
        x_x_bar = images[iter_image] - mean
        covariance += x_x_bar @ x_x_bar

    covariance /= len(images)

    print("covariance", covariance.shape)
    print(covariance)
    eigen_values_unsorted, eigen_vectors_unsorted = linalg.eig(covariance)

    Eigen_index = np.argsort(eigen_values_unsorted)
    Eigen_Value = eigen_values_unsorted[Eigen_index]
    Eigen_Vector = eigen_vectors_unsorted[Eigen_index]

    eigen_vec = (images - mean).T @ Eigen_Vector

    for iter_vector in rnage(len(eigen_vec)):
        eigen_vec[:, iter_vector] /= np.linalg.norm(eigen_vec[:, iter_vector])

    return mean, W

