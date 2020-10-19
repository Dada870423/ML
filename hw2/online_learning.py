import sys
import math
import numpy as np
from math import factorial
class online_learning():
    def __init__(self):
        self.read = False
        self.case = 0
    #    self.train_label_file = train_label_file
    #    self.test_image_file = test_image_file
    #    self.test_label_file = test_label_file


    def read_file(self, file):
        if not self.read:
            self.ptr = open(file, "r")
            line = self.ptr.readline()
            self.read = True
        else:
            line = self.ptr.readline()
        self.case = self.case + 1
        return line

    def Train(self, file):
        self.a = int(input("Input your initial a:"))
        self.b = int(input("Input your initial b:"))
        print("a", self.a, "b", self.b)
        input_test = self.read_file(file = file)
        while input_test:
            print("case ", self.case, ": ", input_test, end = "")
            with_1 = input_test.count("1")
            with_0 = input_test.count("0")

            total = with_1 + with_0
            C_a_b = factorial(total) / factorial(with_1) / factorial(with_0)
            #print(C_a_b)
            probiblity = C_a_b * (with_1 / total) ** with_1
            likelihood = probiblity * (with_0 / total) ** with_0
            
            print("Likelihood:     ", likelihood)
            print("Beta prior:      a = ", self.a, " b = ", self.b)
            self.a = self.a + with_1
            self.b = self.b + with_0
            print("Beta posterior:  a = ", self.a, " b = ", self.b, "\n")           

            input_test = self.read_file(file = file)






