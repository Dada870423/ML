import sys
import math
import numpy as np
from UTIL import *
class MNIST_DISCRETE():

    def __init__(self):
        self.Prior = np.zeros((10), dtype = float) ## Just count it!!
        self.Frequency = np.zeros((10, 28 * 28, 32), dtype = int)
        self.final_image = np.zeros((10, 28 * 28), dtype = int)
        self.trained = False

#    def init_data(self, label_file, image_file):
#        Label_fptr = open(label_file, "rb")
#        Image_fptr = open(image_file, "rb")
#
#        ## init label file
#        Label_fptr.read(4) ## magic number
#        Label_fptr.read(4) ## number of items
#
#        ## init image file
#        Image_fptr.read(4) ## magic number
#        Image_fptr.read(4) ## number of images
#        Image_fptr.read(4) ## number of rows
#        Image_fptr.read(4) ## number of columns
#
#        return Label_fptr, Image_fptr

#    def get_label(self, fptr):
#        label = int.from_bytes(fptr.read(1), byteorder = 'big')
#        return label
#
#    def get_pixel(self, fptr):
#        pixel = int.from_bytes(fptr.read(1), byteorder = 'big')
#        return pixel
#
#    def norm_probability(self, probability):
#        total = 0.0
#        for iter_i in range(len(probability)):
#            total = total + probability[iter_i]
#        for iter_i in range(len(probability)):
#            probability[iter_i] = float(float(probability[iter_i]) / float(total))
#        return probability


    def TRAIN(self, train_label_file, train_image_file):
        Label_fptr, Image_fptr = init_data(label_file = train_label_file, image_file = train_image_file)
        train_case_num = 60000
        printProgress(0, 100, prefix="training:", suffix="Complete", barLength=50)

        for iter_label in range(train_case_num):
            label = get_label(fptr = Label_fptr)
            self.Prior[label] = self.Prior[label] + 1
            self.image_process(label = label, Image_fptr = Image_fptr)
            printProgress(int(iter_label/600), 100, prefix="training: ", suffix="Complete", barLength=50)

        self.Prior = norm_probability(probability = self.Prior)
        self.trained = True
        return self.Prior


    def image_process(self, label, Image_fptr):
        for iter_pixel in range(28 * 28):
            pixel = get_pixel(fptr = Image_fptr)
            self.Frequency[label][iter_pixel][int(pixel / 8)] = \
            	self.Frequency[label][iter_pixel][int(pixel / 8)] + 1

    def Get_Frequency(self):
        if(self.trained):
            return self.Frequency
        else:
            print("not train yet")
            return None

    def cal_image_sum(self):
        Frequency_sum = np.zeros((10, 28 * 28), dtype = float)
        for digit in range(10):
        	for iter_pixel in range(28 * 28):
        		Frequency_sum[digit][iter_pixel] = np.sum(self.Frequency[digit][iter_pixel])

        return Frequency_sum

    def cal_final_image(self):
        for digit in range(10):
            for iter_pixel in range(28 * 28):
                is_1 = np.sum(self.Frequency[digit][iter_pixel][16:32])
                is_0 = np.sum(self.Frequency[digit][iter_pixel][:16])
                if is_1 > is_0:
                    self.final_image[digit][iter_pixel] = 1
                else:
                    self.final_image[digit][iter_pixel] = 0



    def Print_digit(self, label):
        print("label", label)
        for pixel_y in range(28):
            for pixel_x in range(28):
                print(int(np.sum(self.Frequency[label][28 * pixel_y + pixel_x][16:32]) \
                         >= np.sum(self.Frequency[label][28 * pixel_y + pixel_x][:16])), end = "")
            print("")
        print("\n\n\n")

    def get_image(self, ptr):
        image = np.zeros((28 * 28), dtype = float)
        for iter_pixel in range(28 * 28):
            image[iter_pixel] = (get_pixel(fptr = ptr) / 8)
        return image

    def cal_probability(self, test_image, Frequency_sum):
        predict_probability = np.zeros((10), dtype = float)
        for digit in range(10):
        	predict_probability[digit] = np.log(self.Prior[digit])
        	for iter_pixel in range(28 * 28):
        		if self.Frequency[digit][iter_pixel][int(test_image[iter_pixel])] == 0:
        			predict_probability[digit] += \
        				np.log(float(0.00001 / Frequency_sum[digit][iter_pixel]))
        		else:
        			predict_probability[digit] += \
        				np.log(float(float(self.Frequency[digit][iter_pixel][int(test_image[iter_pixel])]) / float(Frequency_sum[digit][iter_pixel])))

        return predict_probability




    def Test(self, test_label_file, test_image_file):
        Label_fptr, Image_fptr = init_data(label_file = test_label_file, image_file = test_image_file)
        Error = 0
        test_case_num = 10000
        Frequency_sum = self.cal_image_sum()
        ErrorRate_fptr = open('MNIST_DISCRETE_ErrorRate.txt', 'w')


        printProgress(0, 100, prefix="testing:", suffix="Complete", barLength=50)
        for test_case in range(test_case_num):
            test_label = get_label(fptr = Label_fptr)
            test_image = self.get_image(ptr = Image_fptr)
            prepre = self.cal_probability(test_image = test_image, Frequency_sum = Frequency_sum)
            predict_probability = norm_probability(probability = prepre)

            Error = Error + compare(predict_probability = predict_probability, Ans = test_label, fptr = ErrorRate_fptr)
            ErrorRate_fptr.write("Error rate: " + str(float(Error / (test_case + 1))))
            printProgress(int(test_case / 100), 100, prefix="testing:", suffix="Complete", barLength=50)
        ErrorRate_fptr.close()


