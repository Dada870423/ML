import numpy as np
import os
from SVM import *

mode = int(input("your mode :  ")) ##Choose which part to run

if mode < 0 or mode >2:
    print("input the valid mode")
    exit(0)

svm = SupportVectorMachine(mode = mode)

if not os.path.isfile("./Train_file.txt"):
    svm.ReadTrainingFile(file = ["X_train.csv", "Y_train.csv"])

if not os.path.isfile("./Test_file.txt"):
    svm.ReadTestFile(file = ["X_test.csv", "Y_test.csv"])

svm.RUN()