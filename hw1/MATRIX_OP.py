import numpy as np
import copy

class MATRIX_OP():
    def iNvErSe(self, input_matrix):
        target_matrix = copy.deepcopy(input_matrix)
        LL = self.Cholesky(input_matrix = target_matrix)
        Inv_L = self.Inv_l(LL = LL)
        Inv_U = Inv_L.T
        inv_input_matrix = Inv_U @ Inv_L

        return inv_input_matrix


    def Inv_l(self, LL):
        m, n = LL.shape
        II = np.identity(n)
        for iter_i in range(m - 1):
            for iter_j in range(1, m - iter_i):
                key = LL[iter_i + iter_j][iter_i] / LL[iter_i][iter_i]
                II[iter_i + iter_j] = II[iter_i + iter_j] - key * II[iter_i]
        for iter_i in range(n):
            if LL[iter_i][iter_i] != 0:
                II[iter_i] = II[iter_i] / LL[iter_i][iter_i]

        return II



    def Cholesky(self, input_matrix):
        m, n = input_matrix.shape
        for iter_i in range(m - 1):
            for iter_k in range(1, m - iter_i):
                key = input_matrix[iter_i + iter_k][iter_i] / input_matrix[iter_i][iter_i]
                if key != 0:
                    for iter_j in range(iter_i, n):
                    	input_matrix[iter_i + iter_k][iter_j] = input_matrix[iter_i + iter_k][iter_j] - key * input_matrix[iter_i][iter_j]
        L = self.GetOutSigma(UUU = input_matrix)
        return L
    def GetOutSigma(self, UUU):
        m, n = UUU.shape
        for iter_i in range(m):
            key = UUU[iter_i][iter_i] ** 0.5
            for iter_j in range(iter_i, n):
                if key != 0:
                    UUU[iter_i][iter_j] = UUU[iter_i][iter_j] / key
        return UUU.T ## return L