import numpy as np
from UTIL import *


class EMEM(object):
    def __init__(self, Binomial_matrix):
        self.input_N = 60000
        self.lamBda = np.ones(10)
        self.probability = np.random.rand(10, 28 * 28)
        self.hidden_W = np.ones((self.input_N, 10))
        self.Binomial_matrix = Binomial_matrix
        self.jimmy = 100 # big number!!




    def E_step(self):
        #print(self.lamBda)
        #print((self.probability[3]))
        #print((self.probability[4]))
        for iter_image in range(self.input_N):
            for iter_digit in range(10):
                self.hidden_W[iter_image][iter_digit] = self.lamBda[iter_digit]
                it = 0
                for iter_pixel in range(28 * 28):
                    if self.Binomial_matrix[iter_image][iter_pixel] == 1:
                        self.hidden_W[iter_image][iter_digit] *= self.probability[iter_digit][iter_pixel]
                    else:
                        self.hidden_W[iter_image][iter_digit] *= (1 - self.probability[iter_digit][iter_pixel])
                    ## deal with underflow
                    if iter_pixel % 10 == 0:
                        self.hidden_W[iter_image][iter_digit] *= self.jimmy

                            

                #print("in Estep")
            #print(self.hidden_W[iter_image])
            #if iter_image == 3:
            #    print("before", self.hidden_W[3])
            #if iter_image == 4:
            #    print("before", self.hidden_W[4])
            self.hidden_W[iter_image] = norm_probability(self.hidden_W[iter_image])
        #print("after", self.hidden_W[3])
        #print("after", self.hidden_W[4])




                


    def M_step(self):
        ## lamBda
        #print(self.lamBda)
        for iter_digit in range(10):
            self.lamBda[iter_digit] = 0
            for iter_image in range(self.input_N):
                self.lamBda[iter_digit] += self.hidden_W[iter_image][iter_digit]
            #self.lamBda[iter_digit] = self.lamBda[iter_digit]


        for iter_digit in range(10):
            for iter_pixel in range(28 * 28):
                self.probability[iter_digit][iter_pixel] = 0
                for iter_image in range(self.input_N):
                    if self.Binomial_matrix[iter_image][iter_pixel] == 1:
                        self.probability[iter_digit][iter_pixel] += self.hidden_W[iter_image][iter_digit]
                            
                self.probability[iter_digit][iter_pixel] /= self.lamBda[iter_digit]
        self.lamBda= norm_probability(self.lamBda)













