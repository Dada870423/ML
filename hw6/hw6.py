import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform
import imageio

from KMEANS import *
from SPECTRAL import *

file = "./image1.png"
Gamma_s = 0.0000001 ## for spatial
Gamma_c = 0.0002 ## for color

image = Read_image(file = file) ## read file

## precompute the kernel
Gram = PreComputed_kernel(image = image, Gamma_s = Gamma_s, Gamma_c = Gamma_c) 

## Using Kmeans clustering
GIF_Kmeans = Kmeans(Gram = Gram, k = 5, mode = 1)
## Using Spectral clustering
GIF_spectral = Spectral(Gram = Gram, k = 5, mode = 1)
## Save Gif
imageio.mimsave("Spectral_eric.gif", GIF_Kmeans)
imageio.mimsave("Spectral_eric.gif", GIF_spectral)