import numpy as np
from UTIL import *
import copy


class EMEM(object):
    def __init__(self, Binomial_matrix):
        self.input_N = 1000
        self.lamBda = np.ones(10)
        self.probability = np.random.rand(28 * 28, 10)
        self.hidden_W = np.ones((self.input_N, 10))
        self.Binomial_matrix = Binomial_matrix
        self.jimmy = 100 # big number!!




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
        ans = np.zeors(10)
        for iter_digit in range(10):
            print("hihi")










