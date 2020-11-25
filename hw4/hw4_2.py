import numpy as np
#import argparse
from gaussian import *
from EMEM import *
from UTIL import *
from MNIST import *

train_label_file = "file/train-labels-idx1-ubyte"
train_image_file = "file/train-images-idx3-ubyte"
train_label_file = "file/train-labels-idx1-ubyte"
train_image_file = "file/train-images-idx3-ubyte"








Binomial_matrix = Get_Binomial(train_image_file = train_image_file)
#print(Binomial_matrix[3])



eMeM = EMEM(Binomial_matrix = Binomial_matrix)
#print(eMeM.lamBda)
#print(np.sum(eMeM.lamBda))
#bio_ptr = open("binomial.txt", "r")
#line = bio_ptr.readlines()

#np.save("bio.txt", line)
#
eMeM.E_step()
print("lambda before", eMeM.lamBda)
eMeM.M_step()
print(eMeM.hidden_W[3])
print(eMeM.hidden_W[4])
print(eMeM.hidden_W[5])
print("lambda", eMeM.lamBda)


eMeM.E_step()
eMeM.M_step()
print(eMeM.hidden_W[3])
print(eMeM.hidden_W[4])
print("lambda", eMeM.lamBda)


eMeM.E_step()
eMeM.M_step()
print(eMeM.hidden_W[3])
print(eMeM.hidden_W[4])
print("lambda", eMeM.lamBda)


#eMeM.E_step()
#eMeM.M_step()
#print(eMeM.hidden_W[3])
#print(eMeM.hidden_W[4])
#print("lambda", eMeM.lamBda)
#
#eMeM.E_step()
#eMeM.M_step()
#print(eMeM.hidden_W[3])
#print(eMeM.hidden_W[4])
#print("lambda", eMeM.lamBda)

#
#eMeM.E_step()
#eMeM.M_step()
#print(eMeM.hidden_W[3])
#print(eMeM.hidden_W[4])
#
#
#eMeM.E_step()
#eMeM.M_step()
#print(eMeM.hidden_W[3])
#print(eMeM.hidden_W[4])
#
#print(eMeM.probability[3])
#
#
#print(eMeM.probability[3][160])
#
for iter_y in range(28):
    for iter_x in range(28):
        print(int(eMeM.probability[iter_x + iter_y * 28][1]), end = " ")
    print()
    #line = bio_ptr.readline()
#for iter_y in range(28):
#    for iter_x in range(28):
#    	if eMeM.probability[1][iter_x + iter_y * 28] < 0.5:
#    		print("0")
#    	else:
#    		print("1")
#    print()
#print("\n\n\n")
#
#
#print("")
#print(int(data))

#data = np.loadtxt("binomial.txt")

#print(int(data[0]))


