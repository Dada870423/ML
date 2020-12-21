from READFILE import *
from PCA import *
import random
import numpy as np



mode = 0
Size = (100, 100)


TrainPath = "./Yale_Face_Database/Training/"
TestPath = "./Yale_Face_Database/Testing/"
images, label = Readfile(path = TrainPath, Size = Size)
test_images, test_label = Readfile(path = TestPath, Size = Size)

print(images.shape)
print(images[1].shape)
#print(label)

print(len(label))

if mode == 0:
    index = random.sample(range(len(test_label)), 10)
    sample_image = test_images[index]
    print(index)
    mean, EigenFace = PCA(images = images, Size = Size, FacePath = "./EigenFace/")

    Reconstruct(EigenFace = EigenFace, sample_image = sample_image, Size = Size)

