import numpy as np

from SVM import *

mode = int(input("your mode :  "))

file = ["X_train.csv​", "Y_train.csv​"]

print("X_train.csv​")


if mode == 0:
    print("comparison")
elif mode == 1:
    print("C-SVC")
elif mode == 2:
    print("user-defined kernel")
else:
    print("input the valid mode")
    exit(0)

svm = SupportVectorMachine(mode = mode)

svm.ReadTrainingFile(file = ["X_train.csv", "Y_train.csv"])

svm.ReadTestFile(file = ["X_test.csv", "Y_test.csv"])

svm.RUN()


svm.print_image(4)


