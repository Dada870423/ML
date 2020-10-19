from MNIST_DISCRETE import MNIST_DISCRETE
import os


MNIST_DS = MNIST_DISCRETE()

MNIST_DS.TRAIN("train-labels-idx1-ubyte", "train-images-idx3-ubyte")




#MNIST_DS.Test(test_label_file = "t10k-labels-idx1-ubyte", test_image_file = "t10k-images-idx3-ubyte")

#MNIST_DS.print_fre()
MNIST_DS.cal_final_image()
for digit in range(10):
    MNIST_DS.Print_digit(label = digit)

