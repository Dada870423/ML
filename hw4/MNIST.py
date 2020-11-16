import numpy as np
from UTIL import *

def Get_Binomial(train_image_file):
	#print("train_image_file")
	input_N = 1000
	binomial_mat = np.zeros((input_N, 28 * 28))
	Image_fptr = init_data(image_file = train_image_file)
	for iter_image in range(input_N):
		for iter_pixel in range(28 * 28):
			pixel = get_pixel(fptr = Image_fptr)
			binomial_mat[iter_image][iter_pixel] = (pixel > 127)
	return binomial_mat



