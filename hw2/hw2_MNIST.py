from MNIST_CONTINUEOUS import MNIST_CONTINUEOUS
import os

CD_mode = input("Continuous or Discrete ?????, your choice is : ")

if CD_mode != "Discrete":
    MNIST_CC = MNIST_CONTINUEOUS()
    print("Continuous gogo!!")
    MNIST_CC.TRAIN("train-labels-idx1-ubyte", "train-images-idx3-ubyte")
    M, V, P = MNIST_CC.Get_MVP()
    MNIST_CC.Test(test_label_file = "t10k-labels-idx1-ubyte", test_image_file = "t10k-images-idx3-ubyte")
    for label in range(10):
        MNIST_CC.Print_digit(label = label)
else:
    MNIST_DS = MNIST_DISCRETE()
    MNIST_DS.TRAIN("train-labels-idx1-ubyte", "train-images-idx3-ubyte")
    MNIST_DS.Test(test_label_file = "t10k-labels-idx1-ubyte", test_image_file = "t10k-images-idx3-ubyte")
    for digit in range(10):
        MNIST_DS.Print_digit(label = digit)


