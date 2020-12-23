from READFILE import *
from PCA import *
from LDA import *
from UTIL import *

import random
import numpy as np



mode = 1
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
    mean, PCA_EigenFace = PCA(images = images, Size = Size, FacePath = "./PCA/EigenFace/")

    Reconstruct(EigenFace = PCA_EigenFace, sample_image = sample_image, Size = Size, Path = "./PCA/")
elif mode == 1: ##LDA 
    mean, EigenFace = LDA(images = images, Size = Size, label = label, FacePath = "./LDA/EigenFace/")
    #cate_mean, EigenFace = LDA(images = images, Size = Size, label = label, FacePath = "./LDA/EigenFace/")
    #Reconstruct(EigenFace = EigenFace, sample_image = sample_image, Size = Size, Path = "./LDA/")








