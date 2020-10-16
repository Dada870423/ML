from MNIST_CONTINUEOUS import MNIST_CONTINUEOUS

MNIST_CC = MNIST_CONTINUEOUS()

MNIST_CC.TRAIN("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte")

M, V, P = MNIST_CC.Get_MVP()

print(P)