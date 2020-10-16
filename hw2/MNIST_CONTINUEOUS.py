import sys
import numpy as np
class MNIST_CONTINUEOUS():
    #def __init__(self, train_image_file, train_label_file, test_image_file, test_label_file):
    #    self.train_image_file = train_image_file
    #    self.train_label_file = train_label_file
    #    self.test_image_file = test_image_file
    #    self.test_label_file = test_label_file

    def __init__(self):
        self.Prior = np.zeros((10), dtype = int) ## Just count it!!
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
            self.Mean[label][iter_pixel] = self.Mean[label][iter_pixel] + pixel
            self.pre_Square[label][iter_pixel] = self.pre_Square[label][iter_pixel] + (pixel * pixel)


    def TRAIN(self, train_label_file, train_image_file):
        Label_fptr, Image_fptr = self.init_data(label_file = train_label_file, image_file = train_image_file)
        #Label_fptr = open(train_label_file, "rb")
        #Image_fptr = open(train_image_file, "rb")
        
        for iter_label in range(60000):
            label = self.get_label(Label_fptr)
            #print("LL", label)
            self.Prior[label] = self.Prior[label] + 1
            self.image_process(label = label, Image_fptr = Image_fptr)
        

        for digit in range(10):
            for iter_pixel in range(28 * 28):
                Mean = self.Mean[digit][iter_pixel]
                pre_Square = self.pre_Square[digit][iter_pixel]
                
                self.Mean[digit][iter_pixel] = float(Mean / self.Prior[digit])
                self.pre_Square[digit][iter_pixel] = float(pre_Square / self.Prior[digit])
                self.Var[digit][iter_pixel] = \
                    self.pre_Square[digit][iter_pixel] - (self.Mean[digit][iter_pixel] ** 2)

                if self.Var[digit][iter_pixel] == 0:
                    self.Var[digit][iter_pixel] = 0.000001
        self.trained = True
        return self.Mean, self.Var, self.Prior

    def Get_MVP(self):
        if(self.trained):
            return self.Mean, self.Var, self.Prior
        else:
            print("not train yet")
            return None, None, None


