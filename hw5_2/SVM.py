import numpy as np
import sys
import csv
from libsvm.svmutil import *



class SupportVectorMachine():
    def __init__(self, mode):
        self.mode = mode
        self.read = 0


    def ReadTrainingFile(self, file):
        image_ptr = open(file[0], "r")
        test_list = list(csv.reader(image_ptr)) ## Use csv.reader function to read the csv file
        list_of_floats = [float(item) for a_list in  test_list  for item in a_list] ## convert string to float
        self.TrainImage = np.array(list_of_floats).reshape(5000, 784) ## reshpare to (5000, 28*28)

        label_ptr = open(file[1], "r")
        test_list = list(csv.reader(label_ptr))
        list_of_floats = [int(item) for a_list in  test_list  for item in a_list] ## convert string to int
        self.TrainLabel = np.array(list_of_floats)

        self.Output_train_file()
        self.read += 1


    def ReadTestFile(self, file):
        image_ptr = open(file[0], "r")
        test_list = list(csv.reader(image_ptr)) ## Use csv.reader function to read the csv file
        list_of_floats = [float(item) for a_list in test_list for item in a_list] ## convert string to float
        self.TestImage = np.array(list_of_floats).reshape(2500, 784) ## reshpare to (5000, 28*28)

        label_ptr = open(file[1], "r")
        test_list = list(csv.reader(label_ptr))
        list_of_floats = [int(item) for a_list in test_list for item in a_list] ## convert string to int
        self.TestLabel = np.array(list_of_floats)

        self.Output_test_file()
        self.read += 10

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
        if self.mode == 0:
            print("Comparison")
            self.compare()
        elif self.mode == 1:
            print("C-SVC")
            self.grid()
        else:
            print("User-defined kernel")
            self.linear_RBF()

    def compare(self):
        train_y, train_x = svm_read_problem("Train_file.txt") ## Read Train_file.txt
        test_y, test_x = svm_read_problem("Test_file.txt") ## Read Test_file.txt

        ## Use svm_train function with linear kernel and get the model.
        linear_model = svm_train(train_y, train_x, '-t 0')
        ## Predict the test set with linear model.
        linear_label, linear_acc, linear_val = svm_predict(test_y, test_x, linear_model) 

        ## Use svm_train function with polynomial kernel and get the model.
        poly_model = svm_train(train_y, train_x, '-t 1')
        ## Predict the test set with polynomial model.
        poly_label, poly_acc, poly_val = svm_predict(test_y, test_x, poly_model)

        ## Use svm_train function with RBF kernel and get the model.
        RBF_model = svm_train(train_y, train_x, '-t 2')
        ## Predict the test set with RBF model.
        RBF_label, RBF_acc, RBF_val = svm_predict(test_y, test_x, RBF_model)

        print ("linear_acc : ", linear_acc[0] )
        print ("poly_acc : ", poly_acc[0])
        print ("RBF_acc : ", RBF_acc[0])

    def grid(self):
        Cost = [2 ** (15 - 2 * i) for i in range(11)] ## cost from 2^15 ~ 2^-5
        Gamma = [2 ** (3 - 2 * i) for i in range(10)] ## gamma from 2^3 ~ 2^-15
        Best_gamma = np.zeros(3)
        Best_cost = np.zeros(3)
        Best_rate = np.zeros(3)   

        train_y, train_x = svm_read_problem("Train_file.txt") ## Read Train_file.txt
        test_y, test_x = svm_read_problem("Test_file.txt") ## Read Test_file.txt

        for kernel in range(3): ## grid in three functions
            for iter_cost in Cost:
                for iter_gamma in Gamma:
                    opt = '-s 0 -t ' + str(kernel) + ' -c ' + str(iter_cost) + ' -g ' + str(iter_gamma) + ' -v 5'
                    acc = svm_train(train_y, train_x, opt)
                    if float(acc) > Best_rate[kernel]:
                        Best_gamma[kernel] = iter_gamma
                        Best_cost[kernel] = iter_cost
                        Best_rate[kernel] = float(acc)
        print ("Gamma : ", Best_gamma)
        print ("Cost : ", Best_cost)
        print ("acc : ", Best_rate)

    def RBF(self, X1, X2): ## function of calculating RBF
        gamma = 1 / (28 * 28)
        norm = np.linalg.norm(X1 - X2)
        distance = norm ** 2

        return np.exp(-gamma * distance)
    
    def linear_RBF(self):
        train_y, train_x = svm_read_problem("Train_file.txt")
        test_y, test_x = svm_read_problem("Test_file.txt")


        if self.read != 11:
            self.ReadTrainingFile(file = ["X_train.csv", "Y_train.csv"])
            self.ReadTestFile(file = ["X_test.csv", "Y_test.csv"])

        linear_kernel = self.TrainImage.dot(self.TrainImage.T)
        My_Kernel = np.hstack((np.arange(1, 5001)[:, None], linear_kernel)) ## label and linear kernel
        col, row = My_Kernel.shape
        for iter_col in range(col): ## add RBF to linear kernel
            for iter_row in range(1, row):
                My_Kernel[iter_col][iter_row] += self.RBF(X1 = self.TrainImage[iter_col], X2 = self.TrainImage[iter_row - 1])

        test_linear_kernel = self.TestImage.dot(self.TrainImage.T) ## test kernel
        Test_Kernel = np.hstack((np.arange(1, 2501)[:, None], test_linear_kernel))
        col, row = Test_Kernel.shape
        for iter_col in range(col):
            for iter_row in range(1, row):
                Test_Kernel[iter_col][iter_row] += self.RBF(X1 = self.TestImage[iter_col], X2 = self.TrainImage[iter_row - 1])

        model = svm_train(train_y, My_Kernel, '-t 4') ## SVM train with precompute kernel
        label, acc, val = svm_predict(test_y, Test_Kernel, model)

        print ("User-defined kernel : ", acc[0])


    def print_image(self, number):
        for i in range(28):
            for j in range(28):
                print(int(self.TrainImage[number][i*28 + j] > 0.5))
            print()
        for i in range(30):
            print(self.TrainLabel[i])