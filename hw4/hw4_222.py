import numpy as np
#import argparse
from gaussian import *


from UTIL import *
from MNIST import *

from fuck_EM import *
train_label_file = "file/train-labels-idx1-ubyte"
train_image_file = "file/train-images-idx3-ubyte"


Binomial_matrix = Get_Binomial(train_image_file = train_image_file)



E_step()
M_step()

E_step()
M_step()

E_step()
M_step()

E_step()
M_step()

E_step()
M_step()

E_step()
M_step()


GroundTruth = Test()


for iter_y in range(10):
	for iter_x in range(10):
		print(int(GroundTruth[iter_y][iter_x]), end = " ")
	print()

for iter_y in range(10):
	print(iter_y, "->   : ", GroundTruth[iter_y].argmax())



