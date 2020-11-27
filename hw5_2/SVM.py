import numpy as np
import sys
import csv
from svmutil import *
#from libsvm.python.svmutil import *
#from libsvm.python.svm import *


class SupportVectorMachine():
    def __init__(self, mode):
        self.mode = mode
        self.input_size = 5000
        #self.TrainImage = np.zeros((self.input_size, 784))


    def ReadTrainingFile(self, file):
        image_ptr = open(file[0], "r")
        test_list = list(csv.reader(image_ptr))
        list_of_floats = [float(item) for a_list in  test_list  for item in a_list]
        self.TrainImage = list(np.array(list_of_floats).reshape((5000, 784)))
        #self.TrainImage = np.array(list_of_floats).astype(np.float64)
        #print(haha)

        label_ptr = open(file[1], "r")
        test_list = list(csv.reader(label_ptr))
        list_of_floats = [int(item) for a_list in  test_list  for item in a_list]
        self.TrainLabel = list(np.array(list_of_floats).astype(np.int32).ravel())



        #self.TrainImage = haha

        #image_ptr = open(file[1], "r")

    def ReadTestFile(self, file):
        image_ptr = open(file[0], "r")
        test_list = list(csv.reader(image_ptr))
        print(np.size(test_list))
        list_of_floats = [float(item) for a_list in  test_list  for item in a_list]
        self.TestImage = np.array(list_of_floats).reshape((2500, 784)).astype(np.float64)
        #self.TestImage = np.array(list_of_floats)

        label_ptr = open(file[1], "r")
        test_list = list(csv.reader(label_ptr))
        list_of_floats = [int(item) for a_list in  test_list  for item in a_list]
        self.TestLabel = np.array(list_of_floats).astype(np.int32).ravel()


    def RUN(self):
        result = 0
        if self.mode == 0:
            result = self.compare()
        return result

    def compare(self):
        param = svm_parameter('-t 0')
        prob  = svm_problem(self.TrainLabel, self.TrainImage)
        m = svm_train(prob, param)
        res = svm_predict(self.TestLabel, self.TestImage, m)
        return res





    def print_image(self, number):
        for i in range(28):
            for j in range(28):
                print(int(self.TrainImage[number][i*28 + j] > 0.5))
            print()
        for i in range(30):
            print(self.TrainLabel[i])




