import numpy as np
import os

from SVM import *

mode = int(input("your mode :  "))

file = ["X_train.csv", "Y_train.csv"]

print("X_train.csv")


if mode == 0:
    print("comparison")
elif mode == 1:
    print("C-SVC")
elif mode == 2:
    print("user-defined kernel")
else:
    print("input the valid mode")
    exit(0)

train_file = "./Train_file.txt"
test_file = "./Test_file.txt"

svm = SupportVectorMachine(mode = mode)

if not os.path.isfile(train_file):
    svm.ReadTrainingFile(file = ["X_train.csv", "Y_train.csv"])
else:
    print("not exist")
    svm.Output_train_file()
if not os.path.isfile(test_file):
    svm.ReadTestFile(file = ["X_test.csv", "Y_test.csv"])
else:
    print("not exist")
    svm.Output_test_file()




