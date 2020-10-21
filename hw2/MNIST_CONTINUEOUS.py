import sys
import math
import numpy as np
from UTIL import *
class MNIST_CONTINUEOUS():


    def __init__(self):
        self.Prior = np.zeros((10), dtype = float) ## Just count it!!
        self.Mean = np.zeros((10, 28 * 28), dtype = float)
        self.Var = np.zeros((10, 28 * 28), dtype = float)
        self.pre_Square = np.zeros((10, 28 * 28), dtype = float)
        self.trained = False

    def image_process(self, label, Image_fptr):
        for iter_pixel in range(28 * 28):
            pixel = get_pixel(fptr = Image_fptr)
            self.Mean[label][iter_pixel] = self.Mean[label][iter_pixel] + pixel
            self.pre_Square[label][iter_pixel] = self.pre_Square[label][iter_pixel] + (pixel * pixel)


    def TRAIN(self, train_label_file, train_image_file):
        Label_fptr, Image_fptr = init_data(label_file = train_label_file, image_file = train_image_file)
        train_case_num = 60000
        printProgress(0, 100, prefix="training:", suffix="Complete", barLength=50)
        for iter_label in range(train_case_num):
            label = get_label(fptr = Label_fptr)
            self.Prior[label] = self.Prior[label] + 1
            self.image_process(label = label, Image_fptr = Image_fptr)
            printProgress(int((iter_label + 1)/600), 100, prefix="training: ", suffix="Complete", barLength=50)
        

        for digit in range(10):
            #print(self.Prior)
            for iter_pixel in range(28 * 28):
                
                self.Mean[digit][iter_pixel] = \
                    float(self.Mean[digit][iter_pixel] / self.Prior[digit])
                
                self.pre_Square[digit][iter_pixel] = \
                    float(self.pre_Square[digit][iter_pixel] / self.Prior[digit])
                
                self.Var[digit][iter_pixel] = \
                    self.pre_Square[digit][iter_pixel] - (self.Mean[digit][iter_pixel] ** 2)

                if self.Var[digit][iter_pixel] == 0:
                    self.Var[digit][iter_pixel] = 10000
                elif self.Var[digit][iter_pixel] < 0:
                    self.Var[digit][iter_pixel] = -(self.Var[digit][iter_pixel])
        
        self.Prior = norm_probability(probability = self.Prior)
        #print("norm_print(self.Prior)", self.Prior)
        self.trained = True
        return self.Mean, self.Var, self.Prior

    def Get_MVP(self):
        if(self.trained):
            return self.Mean, self.Var, self.Prior
        else:
            print("not train yet")
            return None, None, None

    def get_image(self, ptr):
        image = np.zeros((28 * 28), dtype = float)
        for iter_pixel in range(28 * 28):
            image[iter_pixel] = get_pixel(fptr = ptr)
        return image

    def Print_digit(self, label):
        print("label", label)
        for pixel_y in range(28):
            for pixel_x in range(28):
                print(int(self.Mean[label][28 * pixel_y + pixel_x] > 128), end = "")
            print("")
        print("\n\n\n")

    def cal_probability(self, test_image):
        predict_probability = np.zeros((10), dtype = float)
        for digit in range(10):
            predict_probability[digit] = np.log(self.Prior[digit])
            for iter_pixel in range(28 * 28):
                tmp1 = np.log(float(1.0 / math.sqrt(2.0 * math.pi * self.Var[digit][iter_pixel])))
                tmp2 = float(((test_image[iter_pixel] - self.Mean[digit][iter_pixel]) ** 2) / (2 * self.Var[digit][iter_pixel]))
                predict_probability[digit] = predict_probability[digit] + tmp1 - tmp2
        return predict_probability





    def Test(self, test_label_file, test_image_file):
        Label_fptr, Image_fptr = init_data(label_file = test_label_file, image_file = test_image_file)
        Error = 0
        test_case_num = 10000
        ErrorRate_fptr = open('MNIST_CONTINUEOUS_ErrorRate.txt', 'w')


        printProgress(0, 100, prefix="testing:", suffix="Complete", barLength=50)
        for test_case in range(test_case_num):
            test_label = get_label(fptr = Label_fptr)
            test_image = self.get_image(ptr = Image_fptr)
            prepre = self.cal_probability(test_image = test_image)
            predict_probability = norm_probability(probability = prepre)
            
            Error = Error + compare(predict_probability = predict_probability, Ans = test_label, fptr = ErrorRate_fptr)
            ErrorRate_fptr.write("Error rate: " + str(float(Error / (test_case + 1))))
            printProgress(int((test_case + 1)/ 100), 100, prefix="testing:", suffix="Complete", barLength=50)
        ErrorRate_fptr.close()












