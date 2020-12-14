import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform
import imageio

from KMEANS import *
from SPECTRAL import *

file = "./image1.png"
Gamma_s = 0.000005 ## for spatial
Gamma_c = 0.000002 ## for color

image = Read_image(file = file) ## read file

## precompute the kernel
Gram = PreComputed_kernel(image = image, Gamma_s = Gamma_s, Gamma_c = Gamma_c) 

## Using Kmeans clustering
GIF_Kmeans = Kmeans(Gram = Gram, k = 2, mode = 1)
## Using Spectral clustering
GIF_spectral = Spectral(Gram = Gram, k = 2, mode = 0)
## Save Gif
np.save("saved_GIF_Kmeans", GIF_Kmeans)

color_list = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]

m, n, p = GIF_Kmeans.shape


GIF_Kmeans_output = []

for iter_m in range(m):
    GIF_Kmeans_output.append([])
    for iter_n in range(n):
        for iter_p in range(p):
            GIF_Kmeans_output[m].append(color_list[int(GIF_Kmeans[m][n][p])])

GIF_Kmeans_output = np.array(GIF_Kmeans_output).reshape((m, 100, 100, 3))

m, n, p = GIF_spectral.shape


GIF_spectral_output = []

for iter_m in range(m):
    GIF_spectral_output.append([])
    for iter_n in range(n):
        for iter_p in range(p):
            GIF_spectral_output[m].append(color_list[int(GIF_spectral_output[m][n][p])])

GIF_spectral_output = np.array(GIF_spectral_output).reshape((m, 100, 100, 3))




imageio.mimsave("Spectral_sean.gif", GIF_Kmeans_output)
imageio.mimsave("Spectral_sean.gif", GIF_spectral_output)