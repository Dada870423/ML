import numpy as np
from UTIL import *


class EMEM(object):
    def __init__(self, Binomial_matrix):
        self.lamBda = norm_probability(np.random.rand(10))
        self.probability = np.random.rand(10, 28 * 28)
        self.hidden_W = np.ones((100, 10))
        self.Binomial_matrix = Binomial_matrix
        self.jimmy = 100000 # big number!!
        self.input_N = 100



    def E_step(self):
        #print(self.lamBda)
        print(type(self.probability[3][3]))
        for iter_image in range(self.input_N):
            for iter_digit in range(10):
                self.hidden_W[iter_image][iter_digit] = self.hidden_W[iter_image][iter_digit] *\
                    self.lamBda[iter_digit]
                for iter_pixel in range(28 * 28):
                    if self.Binomial_matrix[iter_image][iter_digit] == 1:
                        self.hidden_W[iter_image][iter_digit] = self.hidden_W[iter_image][iter_digit] *\
                            self.probability[iter_digit][iter_pixel]
                    else:
                        self.hidden_W[iter_image][iter_digit] = self.hidden_W[iter_image][iter_digit] *\
                            (1 - self.probability[iter_digit][iter_pixel])
                    ## deal with underflow
                    if iter_pixel % 11 == 0:
                        #print(iter_pixel)
                        self.hidden_W[iter_image][iter_digit] = self.hidden_W[iter_image][iter_digit] *\
                            self.jimmy

                print("in Estep")
                self.hidden_W[iter_image] = norm_probability(self.hidden_W[iter_image])




                


    def M_step(self):
        ## lamBda
        #print(self.lamBda)
        for iter_digit in range(10):
            self.lamBda[iter_digit] = 0
            for iter_image in range(self.input_N):
                self.lamBda[iter_digit] = self.lamBda[iter_digit] + self.hidden_W[iter_image][iter_digit]
            self.lamBda[iter_digit] = self.lamBda[iter_digit] / self.input_N


        for iter_digit in range(10):
            for iter_pixel in range(28 * 28):
                self.probability[iter_digit][iter_pixel] = 0
                for iter_image in range(self.input_N):
                    if self.Binomial_matrix[iter_image][iter_pixel] == 1:
                        self.probability[iter_digit][iter_pixel] = self.probability[iter_digit][iter_pixel] +\
                            self.hidden_W[iter_image][iter_digit]










