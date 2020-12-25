from READFILE import *
from PCA import *
from LDA import *
from UTIL import *

import random
import numpy as np



mode = 4
Size = (50, 50)


TrainPath = "./Yale_Face_Database/Training/"
TestPath = "./Yale_Face_Database/Testing/"
images, label = Readfile(path = TrainPath, Size = Size)
test_images, test_label = Readfile(path = TestPath, Size = Size)



cate = np.unique(np.array(label))

print(cate.shape, cate)
index = random.sample(range(len(test_label)), 10)
sample_image = test_images[index]


if mode == 0: ## PCA
    
    print(index)
    mean, PCA_EigenFace, W = PCA(images = images, Size = Size, FacePath = "./PCA/EigenFace/")

    Reconstruct(EigenFace = PCA_EigenFace, sample_image = sample_image, Size = Size, Path = "./PCA/")
elif mode == 1: ##LDA 
    mean, LDA_EigenFace, W = LDA(images = images, Size = Size, label = label, FacePath = "./newLDA/EigenFace/")
    #cate_mean, EigenFace = LDA(images = images, Size = Size, label = label, FacePath = "./LDA/EigenFace/")
    Reconstruct(EigenFace = LDA_EigenFace, sample_image = sample_image, Size = Size, Path = "./newLDA/")
elif mode == 2: ## PCA KNN
    mean, PCA_EigenFace, W = PCA(images = images, Size = Size, FacePath = None)
    #print(Eigen_Vector.shape) ## 135 * 135
    print("PCA_EigenFace", PCA_EigenFace.shape) ## 2500 * 135
    PCA_KNN(k = 4, images = images, EigenFace = PCA_EigenFace.T, proj_train_image = W, label = label, \
        test_images = test_images, test_label = test_label)
elif mode == 3: ## LDA KNN
    mean, LDA_EigenFace, W = LDA(images = images, Size = Size, label = label, FacePath = None)
    #print(Eigen_Vector.shape) ## 135 * 135
    print("LDA_EigenFace", LDA_EigenFace.shape) ## 2500 * 135
    LDA_KNN(k = 5, images = images, EigenFace = LDA_EigenFace.T, proj_train_image = W, label = label, \
        test_images = test_images, test_label = test_label)
elif mode == 4: ## PCA kernel
    mean, PCA_EigenFace, W = PCA_Kernel(images, Gamma = 0.0001, method = 1)
    PCA_KNN(k = 4, images = images, EigenFace = PCA_EigenFace.T, proj_train_image = W, label = label, \
        test_images = test_images, test_label = test_label)









