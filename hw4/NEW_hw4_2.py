import numpy as np
#import argparse
from gaussian import *


from UTIL import *
from MNIST import *

from New_EM import *
train_label_file = "file/train-labels-idx1-ubyte"
train_image_file = "file/train-images-idx3-ubyte"

input_N = 60000
lamBda = np.ones(10)
probability = np.random.rand(28 * 28, 10)
hidden_W = np.ones((input_N, 10))
#Binomial_matrix = Binomial_matrix
jimmy = 100 # big number!!
Label_fptr = Get_label_fptr(label_file = "file/train-labels-idx1-ubyte")
label = np.zeros((10, 6500))



Binomial_matrix = Get_Binomial(train_image_file = train_image_file)

old_lamBda = np.zeors(10)


for iter_ in range(30):
	E_step(Binomial_matrix, input_N = input_N, lamBda = lamBda, probability = probability, hidden_W = hidden_W)
	M_step(Binomial_matrix, lamBda, hidden_W, probability, input_N)
	print(hidden_W[3])
	print(hidden_W[4])
	print(hidden_W[5])
	print("lambda", lamBda)
	norm = np.linalg.norm(old_lamBda - lamBda)
	old_lamBda = copy.deepcopy(lamBda)
	if iter_ > 5 and norm < 0.0001:
		break



GroundTruth = Test(Binomial_matrix, Label_fptr, label, probability)


for iter_y in range(10):
	for iter_x in range(10):
		print(int(GroundTruth[iter_y][iter_x]), end = " ")
	print()

for iter_y in range(10):
	print(iter_y, "->   : ", GroundTruth[iter_y].argmax())



