import numpy as np
import sys



class A_MATRIX(object):
    def __init__(self, filename, polynomial_Bases):
        self.filename = filename
        self.polynomial_Bases = polynomial_Bases
        #self.print_value()

    def print_value(self):
       print(self.polynomial_Bases)

    def read_file(self):
        fp = open(self.filename, "r")
        x = list()
        y = list()
        lines = fp.readlines()
        for row in lines:
            x.append(float(row.split(",")[0]))
            y.append(float(row.split(",")[1]))

        return x, y
    

    def getA(self):
        x, y = self.read_file()
        print(y)