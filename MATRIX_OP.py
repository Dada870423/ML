import numpy as np
import copy

class MATRIX_OP():
    def iNvErSe(self, input_matrix):
        #print("input_matrix", input_matrix)
        target_matrix = copy.deepcopy(input_matrix)
        LL = self.Cholesky(input_matrix = target_matrix)
        Inv_L = self.Inv_l(LL = LL)
        Inv_U = Inv_L.T
        #print("Inv_L", Inv_L)
        #print("Inv_U", Inv_U)
        inv_input_matrix = Inv_U @ Inv_L
        #print("input_matrix", input_matrix)
        #print("inv_input_matrix", inv_input_matrix)
        #print(inv_input_matrix @ input_matrix)
        #print(np.linalg.inv(input_matrix))
        return inv_input_matrix


    def Inv_l(self, LL):
        m, n = LL.shape
        II = np.identity(n)
        for iter_i in range(m - 1):
            for iter_j in range(1, m - iter_i):
                key = LL[iter_i + iter_j][iter_i] / LL[iter_i][iter_i]
                #print("key: ", key)
                II[iter_i + iter_j] = II[iter_i + iter_j] - key * II[iter_i]
                #print("pre_II:", II)
        for iter_i in range(n):
            if LL[iter_i][iter_i] != 0:
                #key = 1 / II[iter_i][iter_i]
                II[iter_i] = II[iter_i] / LL[iter_i][iter_i]
        #print(II @ LL)
        #print("II:", II)
        #print("ans: ", np.linalg.inv(LL))
        return II



    def Cholesky(self, input_matrix):
        m, n = input_matrix.shape
        for iter_i in range(m - 1):
            for iter_k in range(1, m - iter_i):
                #print("first: ", input_matrix[iter_i + 1][iter_i], "second: ", input_matrix[iter_i][iter_i])
                key = input_matrix[iter_i + iter_k][iter_i] / input_matrix[iter_i][iter_i]
                if key != 0:
                    for iter_j in range(iter_i, n):
                    	#print(key)
                    	input_matrix[iter_i + iter_k][iter_j] = input_matrix[iter_i + iter_k][iter_j] - key * input_matrix[iter_i][iter_j]
                    #print(input_matrix)
        L = self.GetOutSigma(UUU = input_matrix)
        #print("L:\n", L)
        return L
    def GetOutSigma(self, UUU):
        #print("GetOutSigma: ############")
        m, n = UUU.shape
        for iter_i in range(m):
            key = UUU[iter_i][iter_i] ** 0.5
            #print("key: ", key)
            for iter_j in range(iter_i, n):
                if key != 0:
                    UUU[iter_i][iter_j] = UUU[iter_i][iter_j] / key
        #print(UUU)
        return UUU.T ## return L