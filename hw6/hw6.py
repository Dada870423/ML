import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform
import imageio

from KMEANS import *
from SPECTRAL import *

file = "./image1.png"
Gamma_s = 0.00005 ## for spatial
Gamma_c = 0.00002 ## for color

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
#GIF = Kmeans(Gram = Gram, k = 4)
#print(it)
#print("GIF", GIF.shape)

#mean_iter_point = get_init_mean(Gram, 2)

#print(mean_iter_point)

GIF = Spectral(Gram = Gram, k = 2)


imageio.mimsave("GiF_spectral_1.gif", GIF)

print()