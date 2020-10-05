import numpy as np

class MATRIX_OP():
    def iNV(self, input_matrix):
        return ans

    def lLT(self, input_matrix):
        return ans

    def getL(self, input_matrix):
        m, n = input_matrix.shape
        for iter_i in range(m - 1):
            for iter_k in range(1, m - iter_i):
                print("first: ", input_matrix[iter_i + 1][iter_i], "second: ", input_matrix[iter_i][iter_i])
                key = input_matrix[iter_i + iter_k][iter_i] / input_matrix[iter_i][iter_i]
                for iter_j in range(iter_i, n):
                	print(key)
                	input_matrix[iter_i + iter_k][iter_j] = input_matrix[iter_i + iter_k][iter_j] - key * input_matrix[iter_i][iter_j]
                print(input_matrix)