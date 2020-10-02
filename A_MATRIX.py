import numpy as np
import sys



class A_MATRIX(object):
    def __init__(self, filename, polynomial_Bases):
        self.filename = filename
        self.polynomial_Bases = polynomial_Bases
        self.x , self.y = self.read_file()
        #self.print_value()

    def getA(self):
        pre_A = self.poly_base()
        A = np.asarray(pre_A, dtype='float').reshape((-1,1))
        return A
    
    def getB(self):
        B = np.asarray(self.y, dtype='float').reshape((-1,1))
        return B

    def poly_base(self):
        A = list()
        for iter_i in range(len(self.x)):
            tmp = list()
            for iter_j in range(self.polynomial_Bases - 1, -1, -1):
                tmp.append(self.x[iter_i] ** iter_j)
            A.append(tmp)
        # print("############", A)
        return A



    def print_value(self):
       print(self.polynomial_Bases)

    def read_file(self):
        fp = open(self.filename, "r")
        xx = list()
        yy = list()
        lines = fp.readlines()
        for row in lines:
            xx.append(float(row.split(",")[0]))
            yy.append(float(row.split(",")[1]))

        return xx, yy
    

