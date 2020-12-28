from READFILE import *
from PCA import *
from LDA import *
from UTIL import *

import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type = int, default = 0)
input_ = parser.parse_args()


#mode = 1
Size = (50, 50)


images, label = Readfile(path = "./Yale_Face_Database/Training/", Size = Size)
test_images, test_label = Readfile(path = "./Yale_Face_Database/Testing/", Size = Size)


sample_image = test_images[random.sample(range(len(test_label)), 10)]


if input_.mode == 0: 
    ## Doing PCA and get the eigenface and W(dimension reduction)
    PCA_mean, PCA_EigenFace, PCA_W = PCA(images = images, Size = Size, FacePath = "./PCA/EigenFace/")
    Reconstruct(EigenFace = PCA_EigenFace, sample_image = sample_image, Size = Size, Path = "./PCA/")

    ## Doing LDA and get the fisherface and W(dimension reduction)
    LDA_mean, LDA_EigenFace, LDA_W = LDA(images = images, Size = Size, label = label, FacePath = "./LDA/EigenFace/")
    Reconstruct(EigenFace = LDA_EigenFace, sample_image = sample_image, Size = Size, Path = "./LDA/")

elif input_.mode == 1: 
    ## Doing PCA and get the eigenface and W(dimension reduction)
    print("PCA:")
    PCA_mean, PCA_EigenFace, PCA_W = PCA(images = images, Size = Size, FacePath = None)
    ## Using PCA Knn on test image sets, I try to label the test images.
    KNN("PCA", k = 3, images = images, EigenFace = PCA_EigenFace.T, proj_train_image = PCA_W, label = label, \
        test_images = test_images, test_label = test_label)
    print("LDA:")
    
    ## Doing LDA and get the fisherface and W(dimension reduction)
    LDA_mean, LDA_EigenFace, LDA_W = LDA(images = images, Size = Size, label = label, FacePath = None)
    ## Using LDA Knn on test image sets, I try to label the test images.
    KNN("LDA", k = 6, images = images, EigenFace = LDA_EigenFace.T, proj_train_image = LDA_W, label = label, \
        test_images = test_images, test_label = test_label)

elif input_.mode == 2: 
    ## Doing kernel PCA and get the eigenface and W(dimension reduction)
    PCA_mean, PCA_EigenFace, PCA_W = PCA_Kernel(images, Gamma = 0.0001, method = 1)
    KNN("PCA", k = 3, images = images, EigenFace = PCA_EigenFace.T, proj_train_image = PCA_W, label = label, \
        test_images = test_images, test_label = test_label)
    ## Doing kernel LDA and get the fisherface and W(dimension reduction)
    LDA_mean, LDA_EigenFace, LDA_W = LDA_Kernel(images = images, Gamma = 0.0001, label = label, method = 1)
    KNN("LDA", k = 5, images = images, EigenFace = LDA_EigenFace.T, proj_train_image = LDA_W, label = label, \
        test_images = test_images, test_label = test_label)




