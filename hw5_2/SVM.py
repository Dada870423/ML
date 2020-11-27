import numpy as np
import sys
import csv
#from svmutil import *
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
        self.TrainImage = list(np.array(list_of_floats))
        #self.TrainImage = np.array(list_of_floats).astype(np.float64)
        #print(haha)

        label_ptr = open(file[1], "r")
        test_list = list(csv.reader(label_ptr))
        list_of_floats = [int(item) for a_list in  test_list  for item in a_list]
        self.TrainLabel = np.array(list_of_floats)



        #self.TrainImage = haha

        #image_ptr = open(file[1], "r")

    def ReadTestFile(self, file):
        image_ptr = open(file[0], "r")
        test_list = list(csv.reader(image_ptr))
        print(np.size(test_list))
        list_of_floats = [float(item) for a_list in  test_list  for item in a_list]
        self.TestImage = np.array(list_of_floats)
        #self.TestImage = np.array(list_of_floats)

        label_ptr = open(file[1], "r")
        test_list = list(csv.reader(label_ptr))
        list_of_floats = [int(item) for a_list in  test_list  for item in a_list]
        self.TestLabel = np.array(list_of_floats)

    def Output_txt(self):
        Train_out_ptr = open("Train.txt", "w")
        for iter_ in range(3):
            line = str(self.TrainLabel[iter_])
            for iter_pixel in range(28 * 28):
                line += " " + str(iter_pixel) + ":" + str(self.TrainImage)
            line += "\n"
            Train_out_ptr.write(line)
        f.close()




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




