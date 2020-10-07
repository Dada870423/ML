import numpy as np
import sys
from MATRIX_OP import MATRIX_OP


class A_MATRIX(object):
    def __init__(self, filename, polynomial_Bases):
        self.filename = filename
        self.polynomial_Bases = polynomial_Bases
        self.x , self.y = self.read_file()

    def getA(self):
        pre_A = self.poly_base()
        A = np.asarray(pre_A, dtype='float')
        return A
    
    def getB(self):
        B = np.asarray(self.y, dtype='float')
        return B

    def getX(self):
        return self.x


    def poly_base(self):
        A = list()
        for iter_i in range(len(self.x)):
            tmp = list()
            for iter_j in range(self.polynomial_Bases - 1, -1, -1):
                tmp.append(self.x[iter_i] ** iter_j)
            A.append(tmp)
        return A


    def read_file(self):
        fp = open(self.filename, "r")
        xx = list()
        yy = list()
        lines = fp.readlines()
        for row in lines:
            xx.append(float(row.split(",")[0]))
            yy.append(float(row.split(",")[1]))

        return xx, yy
    

