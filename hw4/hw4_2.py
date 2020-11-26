import numpy as np
#import argparse
from gaussian import *


from UTIL import *

from MNIST import *
from EMEM import *
train_label_file = "file/train-labels-idx1-ubyte"
train_image_file = "file/train-images-idx3-ubyte"









Binomial_matrix = Get_Binomial(train_image_file = train_image_file)
#print(Binomial_matrix[3])



eMeM = EMEM(Binomial_matrix = Binomial_matrix, train_label_file = train_label_file)
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

for iter_y in range(28):
    for iter_x in range(28):
        print(eMeM.probability[iter_x + iter_y * 28][1], end = " ")
    print()


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
#for iter_y in range(28):
#    for iter_x in range(28):
#        print(eMeM.probability[iter_x + iter_y * 28][1], end = " ")
#    print()
#    #line = bio_ptr.readline()
#for i in range(10):
#	for iter_y in range(28):
#	    for iter_x in range(28):
#	    	if eMeM.probability[iter_x + iter_y * 28][i] < 0.4:
#	    		print("0", end = "")
#	    	else:
#	    		print("1", end = "")
#	    print()
#	print("\n\n\n")


GroundTruth = eMeM.Test()


for iter_y in range(10):
	for iter_x in range(10):
		print(int(GroundTruth[iter_y][iter_x]), end = " ")
	print()

for iter_y in range(10):
	print(iter_y, "->   : ", GroundTruth[iter_y].argmax())


#
#
#print("")
#print(int(data))

#data = np.loadtxt("binomial.txt")

#print(int(data[0]))


