import sys
import math
import numpy as np
class MNIST_CONTINUEOUS():
    #def __init__(self, train_image_file, train_label_file, test_image_file, test_label_file):
    #    self.train_image_file = train_image_file
    #    self.train_label_file = train_label_file
    #    self.test_image_file = test_image_file
    #    self.test_label_file = test_label_file

    def __init__(self):
        self.Prior = np.zeros((10), dtype = float) ## Just count it!!
        self.Mean = np.zeros((10, 28 * 28), dtype = float)
        self.Var = np.zeros((10, 28 * 28), dtype = float)
        self.pre_Square = np.zeros((10, 28 * 28), dtype = float)
        self.trained = False

    def init_data(self, label_file, image_file):
        Label_fptr = open(label_file, "rb")
        Image_fptr = open(image_file, "rb")

        ## init label file
        Label_fptr.read(4) ## magic number
        Label_fptr.read(4) ## number of items

        ## init image file
        Image_fptr.read(4) ## magic number
        Image_fptr.read(4) ## number of images
        Image_fptr.read(4) ## number of rows
        Image_fptr.read(4) ## number of columns

        return Label_fptr, Image_fptr

    def get_label(self, fptr):
        label = int.from_bytes(fptr.read(1), byteorder = 'big')
        #print(label)
        return label

    def get_pixel(self, fptr):
        pixel = int.from_bytes(fptr.read(1), byteorder = 'big')
        return pixel

    def image_process(self, label, Image_fptr):
        #print("ll:", label)
        for iter_pixel in range(28 * 28):
            pixel = self.get_pixel(fptr = Image_fptr)
            #print(pixel)
            self.Mean[label][iter_pixel] = self.Mean[label][iter_pixel] + pixel
            self.pre_Square[label][iter_pixel] = self.pre_Square[label][iter_pixel] + (pixel * pixel)
        #print("nononono")
        #for i in range(len(self.pre_Square)):
        #    print("######", i, self.pre_Square[i])
        #print("square", self.pre_Square)


    def norm_probability(self, probability):
        total = 0.0
        #print(probability)
        for iter_i in range(len(probability)):
            total = total + probability[iter_i]
        #print(probability, "total", total)
        for iter_i in range(len(probability)):
            probability[iter_i] = float(float(probability[iter_i]) / float(total))
        #print(probability)
        return probability


    def TRAIN(self, train_label_file, train_image_file):
        Label_fptr, Image_fptr = self.init_data(label_file = train_label_file, image_file = train_image_file)
        train_case_num = 60000
        for iter_label in range(train_case_num):
            label = self.get_label(Label_fptr)
            #print("LL", label)
            self.Prior[label] = self.Prior[label] + 1
            self.image_process(label = label, Image_fptr = Image_fptr)
        

        for digit in range(10):
            for iter_pixel in range(28 * 28):
                #Mean = self.Mean[digit][iter_pixel]
                #pRe_Square = self.pre_Square[digit][iter_pixel]
                
                self.Mean[digit][iter_pixel] = \
                    float(self.Mean[digit][iter_pixel] / self.Prior[digit])
                #self.pre_Square[digit][iter_pixel] = \
                #    float(self.pre_Square[digit][iter_pixel] / self.Prior[digit])
                self.Var[digit][iter_pixel] = \
                    math.sqrt(self.pre_Square[digit][iter_pixel] - (self.Mean[digit][iter_pixel] ** 2))

                if self.Var[digit][iter_pixel] == 0:
                    self.Var[digit][iter_pixel] = 0.00001
                elif self.Var[digit][iter_pixel] < 0:
                    self.Var[digit][iter_pixel] = -(self.Var[digit][iter_pixel])
        
        #print("fuck", self.Prior / 60000)
        #print(self.Mean)
        #print(self.pre_Square)
        #self.Prior = self.Prior / 60000
        #print(self.Mean[0])
        self.Prior = self.norm_probability(self.Prior)
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
            image[iter_pixel] = self.get_pixel(fptr = ptr)
        return image


    def cal_probability(self, test_image):
        predict_probability = np.zeros((10), dtype = float)
        for digit in range(10):
            predict_probability[digit] = predict_probability[digit] + np.log(self.Prior[digit])
            for iter_pixel in range(28 * 28):
                tmp1 = np.log(float(1.0 / math.sqrt(2.0 * math.pi * self.Var[digit][iter_pixel])))
                tmp2 = float(((test_image[iter_pixel] - self.Mean[digit][iter_pixel]) ** 2) / (2 * self.Var[digit][iter_pixel]))
                predict_probability[digit] = predict_probability[digit] + tmp1 - tmp2
        return predict_probability





    def Test(self, test_label_file, test_image_file):
        Label_fptr, Image_fptr = self.init_data(label_file = test_label_file, image_file = test_image_file)
        Error = 0
        test_case_num = 10000
        for test_case in range(test_case_num):
            test_label = self.get_label(Label_fptr)
            #predict_probability = np.zeros((10), dtype = float)
            test_image = self.get_image(ptr = Image_fptr)
            prepre = self.cal_probability(test_image = test_image)

            #print("before: ", prepre)

            predict_probability = self.norm_probability(prepre)
            
            #print("after:", predict_probability)

            prediction = np.argmin(predict_probability)
            print("Prediction: ", prediction, ", Ans: ", test_label)
            if prediction != test_label:
                Error = Error + 1
        
            print("Posterior (in log scale):")
            for j in range(10):
                print(j, ": ", predict_probability[j])
            print("Error rate: ", float(Error / (test_case + 1)))


    def TTTTTTest(self, M, V, P, test_label_file, test_image_file):
        Label_fptr, Image_fptr = self.init_data(label_file = test_label_file, image_file = test_image_file)
        Error = 0

        for test_case in range(1):
            test_label = self.get_label(Label_fptr)
            print(test_label)
            #predict_probability = np.zeros((10), dtype = float)
            test_image = self.get_image(ptr = Image_fptr)
            predict_probability = self.norm_probability(self.cal_probability(test_image = test_image))
            prediction = np.argmin(predict_probability)
            print("Prediction: ", prediction, ", Ans: ", test_label)
            if prediction != test_label:
                Error = Error + 1
        
        print("Posterior (in log scale):")
        for j in range(10):
            print(j, ": ", predict_probability[j])
        print("Error rate: ", float(Error / 10000))











