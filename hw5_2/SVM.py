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
        self.TrainImage = np.array(list_of_floats).reshape(5000, 784)
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
        self.TestImage = np.array(list_of_floats).reshape(2500, 784)
        #self.TestImage = np.array(list_of_floats)

        label_ptr = open(file[1], "r")
        test_list = list(csv.reader(label_ptr))
        list_of_floats = [int(item) for a_list in  test_list  for item in a_list]
        self.TestLabel = np.array(list_of_floats)

    def Output_train_file(self):
        Train_out_ptr = open("Train_file.txt", "w")
        for iter_image in range(5000):
            line = str(self.TrainLabel[iter_image])
            for iter_pixel in range(28 * 28):
                line += " " + str(iter_pixel) + ":" + str(self.TrainImage[iter_image][iter_pixel])
            line += "\n"
            Train_out_ptr.write(line)
        Train_out_ptr.close()

    def Output_test_file(self):
        Test_out_ptr = open("Test_file.txt", "w")
        for iter_image in range(2500):
            line = str(self.TestLabel[iter_image])
            for iter_pixel in range(28 * 28):
                line += " " + str(iter_pixel) + ":" + str(self.TestImage[iter_image][iter_pixel])
            line += "\n"
            Test_out_ptr.write(line)
        Test_out_ptr.close()


    def RUN(self):
        result = 0
        if self.mode == 0:
            self.compare()
        elif self.mode == 1:
            self.grid()
        else:
            self.linear_RBF()
        return result

    def compare(self):
        #linear_param = svm_parameter('-t 0')
        #poly_param = svm_parameter('-t 1')
        #RBF_param = svm_parameter('-t 2')
        train_y, train_x = svm_read_problem("Train_file.txt")
        test_y, test_x = svm_read_problem("Test_file.txt")

        linear_model = svm_train(train_y, train_x, '-t 0')
        linear_label, linear_acc, linear_val = svm_predict(test_y, test_x, linear_model)


        poly_model = svm_train(train_y, train_x, '-t 1')
        poly_label, poly_acc, poly_val = svm_predict(test_y, test_x, poly_model)

        RBF_model = svm_train(train_y, train_x, '-t 2')
        RBF_label, RBF_acc, RBF_val = svm_predict(test_y, test_x, RBF_model)

        print "linear_acc : ", linear_acc[0] 
        print "poly_acc : ", poly_acc[0]
        print "RBF_acc : ", RBF_acc[0]

    def grid(self):
        Cost = [2 ** (15 - 2 * i) for i in range(11)]
        Gamma = [2 ** (3 - 2 * i) for i in range(10)]
        Best_gamma = np.zeros(3)
        Best_cost = np.zeros(3)
        Best_rate = np.zeros(3)   

        train_y, train_x = svm_read_problem("Train_file.txt")
        test_y, test_x = svm_read_problem("Test_file.txt")

        for kernel in range(3):
            for iter_cost in Cost:
                for iter_gamma in Gamma:
                    opt = '-s 0 -t ' + str(kernel) + ' -c ' + str(iter_cost) + ' -g ' + str(iter_gamma) + ' -v 5'
                    acc = svm_train(train_y, train_x, opt)
                    #label, acc, val = svm_predict(test_y, test_x, model)
                    if float(acc) > Best_rate[kernel]:
                        Best_gamma[kernel] = iter_gamma
                        Best_cost[kernel] = iter_cost
                        Best_rate[kernel] = float(acc)
        print "Gamma : ", Best_gamma
        print "Cost : ", Best_cost
        print "acc : ", Best_rate


    def linear_RBF(self):
        gamma = 1 / (28 * 28)

        train_y, train_x = svm_read_problem("Train_file.txt")
        test_y, test_x = svm_read_problem("Test_file.txt")

        self.ReadTrainingFile(file = ["X_train.csv", "Y_train.csv"])
        self.ReadTestFile(file = ["X_test.csv", "Y_test.csv"])

        #np_train_x = np.array(train_x)
        #np_train_y = np.array(train_y)
        #np_test_y = np.array(test_y)
        #np_test_x = np.array(test_x)

        linear_kernel = self.TrainImage.dot(self.TrainImage.T)

        My_Kernel = np.hstack((np.arange(1, 5001)[:, None], linear_kernel))
        print(My_Kernel.shape)
        #for iter_y in range()







    def print_image(self, number):
        for i in range(28):
            for j in range(28):
                print(int(self.TrainImage[number][i*28 + j] > 0.5))
            print()
        for i in range(30):
            print(self.TrainLabel[i])




