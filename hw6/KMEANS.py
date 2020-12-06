import numpy as np
import cv2
import random
from scipy.spatial.distance import pdist, squareform
import copy

### good parameter for kmeans & kmeans++
## Gamma_s = 0.0000001 ## for spatial
## Gamma_c = 0.0002 ## for color

def Read_image(file):    
    image = cv2.imread(file)
    height, width, C = image.shape
    image_10000_3 = image.reshape(height * width, C)

    return image_10000_3


def Kmeans(Gram, k, mode):
    mean = get_init_mean(Gram = Gram, k = k, mode = mode)

    error = 100
    classify = np.zeros(len(Gram), int)
    distance = np.zeros(k)
    New_mean = np.zeros((k, Gram.shape[1]))
    GIF = np.zeros((1, 100, 100))
    it = 0
    while error > 1:
        it += 1
        ## E-step
        for iter_pixel in range(len(Gram)):
            for iter_k in range(k):
                distance[iter_k] = np.linalg.norm(Gram[iter_pixel] - mean[iter_k])
            classify[iter_pixel] = distance.argmin()

        ## M-step
        
        for iter_k in range(k):
            print(mean.shape)
            In_k_cluster = Gram[classify == iter_k]
            print("In_k_cluster", iter_k, In_k_cluster.shape)
            New_mean[iter_k] = In_k_cluster.sum(axis = 0)
            if len(In_k_cluster) > 0:
                New_mean[iter_k] /= len(In_k_cluster)

        error = np.linalg.norm(New_mean - mean)
        mean = copy.deepcopy(New_mean)
        GIF = np.vstack((GIF, classify.reshape(1, 100, 100)))
        print(error)
        for iter_pixel in range(10000):
            if iter_pixel % 100 == 0:
                print()
            print(int(classify[iter_pixel]), end = "")
        print("\n\n")
        if it > 10:
            break
    return GIF

        







def get_init_mean(Gram, k, mode = 0):
    mean_iter_point = np.zeros(k, int)
    ## kmeans
    if mode == 0:
        iter_ = 0
        while iter_ < k:
            point = random.randint(1, len(Gram))
            print("point", point)
            if not point in mean_iter_point:
                mean_iter_point[iter_] = point
                iter_ = iter_ + 1
        mean_iter_point -= 1

    elif mode == 1: ## kmeans++
        print("kmean++++++++++++")
        #Center = np.zeros((k, Gram.shape[1]))
        mean_iter_point[0] = random.randint(0, len(Gram) - 1)
        #Center[0] = Gram[mean_iter_point[0]]
        for iter_center in range(1, k):
            center_distance = np.zeros(len(Gram))
            classify = np.zeros(len(Gram))
            closest_dis = np.zeros(k)
            for iter_pixel in range(len(Gram)):
                for iter_k in range(iter_center):
                    closest_dis[iter_k] = np.linalg.norm(Gram[iter_pixel] - Gram[mean_iter_point[iter_k]])
                classify[iter_pixel] = closest_dis.argmin()

                center_distance[iter_pixel] = np.linalg.norm(Gram[iter_pixel] - Gram[int(mean_iter_point[int(classify[iter_pixel])])])
            SSUUMM = np.sum(center_distance)
            target = random.uniform(0, SSUUMM)
            for iter_dis in range(len(center_distance)):
                target = target - center_distance[iter_dis]
                if target <= 0:
                    mean_iter_point[iter_center] = iter_dis
                    print("kmean++", mean_iter_point)
                    break
    return Gram[mean_iter_point]

    







def PreComputed_kernel(image, Gamma_s, Gamma_c):
    ## Spatial information, coordinate of pixel (100, 100, 2)
    ## [x, y], ex: Spatial[0][0] = [0, 0], Spatial[1][3] = [1, 3]
    row, color = image.shape
    Spatial = np.zeros((row, 2))
    for iter_y in range(row):
        Spatial[iter_y] = [iter_y // row, iter_y % row]

    ## size = ((row * (row-1)) // 2)
    Kernel_S = (- Gamma_s * pdist(Spatial, 'sqeuclidean'))

    Kernel_C = (- Gamma_c * pdist(image, 'sqeuclidean'))
    #print(Kernel_C)

    Kernelone = np.exp(Kernel_S + Kernel_C)

    Kernel = squareform(Kernelone)
    #print("one", Kernel.shape)
    #for i in range(len(Kernelone)):
    #    print(Kernelone[i], end = " ")

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