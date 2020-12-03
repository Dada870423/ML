import numpy as np
import cv2
import random
from scipy.spatial.distance import pdist, squareform


def Read_image(file):    
    image = cv2.imread(file)
    height, width, C = image.shape
    image_10000_3 = image.reshape(height * width, C)

    return image_10000_3


def Kmeans(Gram, k):
    mean = get_init_mean(Gram = Gram, k = k)

    error = 100
    classify = np.zeros(len(Gram))
    distance = np.zeros(k, int)
    while error < 10:
        ## E-step
        for iter_pixel in range(len(Gram)):
            for iter_k in range(k):
                distance[iter_k] = np.linalg.norm(Gram[iter_pixel] - mean[iter_k])
            classify[iter_pixel] = distance.argmin()

        ## M-step
        New_mean = np.zeros((k, Gram.shape[1]))
        for iter_k in range(k):
            In_k_cluster = Gram[classify == iter_k]
            New_mean[iter_k] = In_k_cluster.sum(axis = 0)
            if len(In_k_cluster) > 0:
                New_mean[iter_k] /= len(In_k_cluster)

        error = #???

        







def get_init_mean(Gram, k):
    mean_iter_point = np.zeros(k, int)
    iter_ = 0
    while iter_ < k:
        point = random.randint(1, len(Gram))
        print("point", point)
        if not point in mean_iter_point:
            mean_iter_point[iter_] = point
            iter_ = iter_ + 1

    mean_iter_point -= 1


    return Gram[mean_iter_point]





def PreComputed_kernel(image, Gamma_s, Gamma_c):
    ## Spatial information, coordinate of pixel (100, 100, 2)
    ## [x, y], ex: Spatial[0][0] = [0, 0], Spatial[1][3] = [1, 3]
    row, color = image.shape
    Spatial = np.zeros((row, 2))
    for iter_y in range(row):
        Spatial[iter_y] = [iter_y // row, iter_y % row]

    ## size = ((row * (row-1)) // 2)
    Kernel_S = np.exp(- Gamma_s * pdist(Spatial,'sqeuclidean'))

    Kernel_C = np.exp(- Gamma_c * pdist(image,'sqeuclidean'))

    Kernelone = (Kernel_S * Kernel_C)

    Kernel = squareform(Kernelone)
    print("one", Kernelone.shape)

    return Kernel

#    it = 0

#    for iter_y in range(1, row):
#        for iter_x in range(iter_y):
#            Kernel_S[it] = RBF(X1 = Spatial[iter_y], X2 = [iter_x], gamma = Gamma_s)
#            #print(it, end = " ")
#            #print("(", iter_y, ", ", iter_x, ")", end = " ")
#            it = it + 1
#        #print()
#
#    print(Kernel_S.shape)



def RBF(X1, X2, gamma): ## function of calculating RBF
    norm = np.linalg.norm(X1 - X2)
    distance = norm ** 2

    return np.exp(-gamma * distance)