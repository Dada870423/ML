from READFILE import *
from PCA import *
import random
import numpy as np



mode = 0


path = "./Yale_Face_Database/Training/"
images, label = Readfile(path)

print(images[1].shape)
#print(label)

print(len(label))

if mode == 0:
    index = random.sample(range(len(label)), 10)
    print(index)
    mean, W = PCA(images = images)

