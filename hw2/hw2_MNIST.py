from MNIST_CONTINUEOUS import MNIST_CONTINUEOUS
from MNIST_DISCRETE import MNIST_DISCRETE
import os

CD_mode = input("Continuous or Discrete ?????, your choice is : ")

if CD_mode != "Discrete":
    MNIST_CC = MNIST_CONTINUEOUS()
    print("Continuous gogo!!")
    MNIST_CC.TRAIN(train_label_file = "file/train-labels-idx1-ubyte", train_image_file = "file/train-images-idx3-ubyte")
    MNIST_CC.Test(test_label_file = "file/t10k-labels-idx1-ubyte", test_image_file = "file/t10k-images-idx3-ubyte")
    for digit in range(10):
        MNIST_CC.Print_digit(label = digit)
else:
    MNIST_DS = MNIST_DISCRETE()
    print("Discrete gogo!!")
    MNIST_DS.TRAIN(train_label_file = "train-labels-idx1-ubyte", train_image_file = "file/train-images-idx3-ubyte")
    MNIST_DS.Test(test_label_file = "file/t10k-labels-idx1-ubyte", test_image_file = "file/t10k-images-idx3-ubyte")
    for digit in range(10):
        MNIST_DS.Print_digit(label = digit)


