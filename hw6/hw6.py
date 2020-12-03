import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform

from KMEANS import *

file = "./image1.png"
Gamma_s = 0.001 ## for spatial
Gamma_c = 0.001 ## for color

#test_123 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])




#
image = Read_image(file = file)
#print(image.shape)
#
Gram = PreComputed_kernel(image = image, Gamma_s = Gamma_s, Gamma_c = Gamma_c)
##PreComputed_kernel(image, Gamma_s, Gamma_c)
#
#
#
#
#print(Gram.shape)
Kmeans(Gram = Gram, k = 2)

#mean_iter_point = get_init_mean(Gram, 2)

#print(mean_iter_point)


print(Gram.shape)
print(len(Gram))


print()