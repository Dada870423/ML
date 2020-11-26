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
label = np.zeros((10, 6800))



Binomial_matrix = Get_Binomial(train_image_file = train_image_file)

old_lamBda = np.zeros(10)


for iter_ in range(30):
	E_step(Binomial_matrix, input_N = input_N, lamBda = lamBda, probability = probability, hidden_W = hidden_W)
	M_step(Binomial_matrix, lamBda, hidden_W, probability, input_N)
	print_P(probability)
	
	norm = np.linalg.norm(old_lamBda - lamBda)
	print("No. of Iteration: ", iter_, " , Difference:",  norm, "\n")
	print("-" * 73)
	old_lamBda = copy.deepcopy(lamBda)
	if iter_ > 5 and norm < 0.001:
		break



GroundTruth = Test(Binomial_matrix, Label_fptr, label, probability)


for iter_y in range(10):
	for iter_x in range(10):
		print(int(GroundTruth[iter_y][iter_x]), end = " ")
	print()

print("GTTTT", GroundTruth)
print("GT.argmax", GroundTruth.argmax())

tmp = copy.deepcopy(GroundTruth)
RRRow = np.zeros(10)
CCCol = np.zeros(10)

for i in range(10):
    now = tmp.argmax()
    row = int(now / 10)
    col = now % 10
    print(row, "  --> ", col)
    RRRow[i] = row
    CCCol[i] = col
    tmp[row] = 0
    tmp[:, col] = 0


for i in range(10):
    RrR = (np.where(RRRow == i))[0][0]
    CcC = int(CCCol[RrR])
    A = GroundTruth[i][CcC]
    B = GroundTruth[:, CcC].sum() - A
    C = GroundTruth[i].sum() - A
    D = 60000 - (B + C) + A
    print("Confusion Matrix", i, ":")
    print("Confusion Matrix:\n                Predict number ", i, "   Predict cluster ", i)
    print("Is number    ", i, "       ", A, "                  ", B)
    print("Isn't number ", i, "       ", C, "                  ", D, "\n")
    print("Sensitivity (Successfully predict cluster 1): ", A / (A + B))
    print("Specificity (Successfully predict cluster 2): ", D / (C + D))









