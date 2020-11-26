import numpy as np
from UTIL import *
import copy


class EMEM(object):
    def __init__(self, Binomial_matrix, train_label_file):
        self.input_N = 1000
        self.lamBda = np.ones(10)
        self.probability = np.random.rand(28 * 28, 10)
        self.hidden_W = np.ones((self.input_N, 10))
        self.Binomial_matrix = Binomial_matrix
        self.jimmy = 100 # big number!!
        self.Label_fptr = Get_label_fptr(label_file = train_label_file)
        self.label = np.zeros((10, 100))




    def E_step(self):
        for iter_image in range(self.input_N):
            for iter_digit in range(10):
                self.hidden_W[iter_image][iter_digit] = copy.deepcopy(self.lamBda[iter_digit])
        for iter_image in range(self.input_N):
            for iter_pixel in range(28 * 28):
                for iter_digit in range(10):
                
                    if self.Binomial_matrix[iter_image][iter_pixel] == 1:
                        self.hidden_W[iter_image][iter_digit] *= self.probability[iter_pixel][iter_digit]
                    else:
                        self.hidden_W[iter_image][iter_digit] *= (1 - self.probability[iter_pixel][iter_digit])
                    ## deal with underflow
                if iter_pixel % 10 == 0:
                    self.hidden_W[iter_image] = norm_probability(self.hidden_W[iter_image])
                    #self.hidden_W[iter_image][iter_digit] *= self.jimmy
            self.hidden_W[iter_image] = copy.deepcopy(norm_probability(self.hidden_W[iter_image]))

            
        self.hidden_W[self.hidden_W<0.001] = 0.001



                


    def M_step(self):
        for iter_digit in range(10):
            self.lamBda[iter_digit] = 0.0
            for iter_image in range(self.input_N):
                self.lamBda[iter_digit] += self.hidden_W[iter_image][iter_digit]


        for iter_digit in range(10):
            for iter_pixel in range(28 * 28):
                self.probability[iter_pixel][iter_digit] = 0.0
        

        for iter_pixel in range(28 * 28):
            for  iter_digit in range(10):
                for iter_image in range(self.input_N):
                    if self.Binomial_matrix[iter_image][iter_pixel] == 1:
                        self.probability[iter_pixel][iter_digit] += (self.hidden_W[iter_image][iter_digit])
                        #print(self.hidden_W[iter_image][iter_digit], self.probability[iter_pixel][iter_digit], iter_pixel, iter_digit)
                            
                
                self.probability[iter_pixel][iter_digit] /= self.lamBda[iter_digit]

        


        self.lamBda= norm_probability(self.lamBda)


    def Test(self):
        GroundTruth = np.zeros((10, 10))
        items = self.Get_label_100()
        for iter_digit in range(10):
            for iter_item in range(int(items[iter_digit])):
                ans = self.Cal_w(image_th = int(self.label[iter_digit][iter_item]))
                print("-- ", iter_digit, " --   : ", ans.argmax())
                GroundTruth[iter_digit][ans.argmax()] += 1
        return GroundTruth



    def Cal_w(self, image_th):
        ans = np.ones(10)

        for iter_pixel in range(28 * 28):
            for iter_digit in range(10):
            
                if self.Binomial_matrix[image_th][iter_pixel] == 1:
                    ans[iter_digit] *= self.probability[iter_pixel][iter_digit]
                else:
                    ans[iter_digit] *= (1 - self.probability[iter_pixel][iter_digit])
                ## deal with underflow
            if iter_pixel % 10 == 0:
                ans = norm_probability(ans)
                #self.hidden_W[iter_image][iter_digit] *= self.jimmy
        ans = norm_probability(ans)
        return ans




    def Get_label_100(self):
        items = np.zeros(10)
        for iter_label in range(100):
            label_now = get_label(self.Label_fptr)
            xxx = int(items[label_now])
            self.label[label_now][xxx] = iter_label
            items[label_now] = items[label_now] + 1
        for iter_digit in range(10):
            for iter_item in range(int(items[iter_digit])):
                print(self.label[iter_digit][iter_item], end = " ")
            print("--------  ", iter_digit, "\n")
        print("items", items)

        return items











