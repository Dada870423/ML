import sys
import csv
import numpy as np  
import matplotlib.pyplot as plt 
import math
import os

from A_MATRIX import A_MATRIX
from LSE import LSE
from NEWTON import NEWTON
from MATRIX_OP import MATRIX_OP
from OUTPUT_line_ans import OUTPUT_line_ans

filename = sys.argv[1]
polynomial_Bases = int(sys.argv[2])
LambDa = float(sys.argv[3])

## get matrix A & b
a_matrix = A_MATRIX(filename = filename, polynomial_Bases = polynomial_Bases)
A = a_matrix.getA()
b = a_matrix.getB()
input_x = a_matrix.getX()


## LSE
Lse = LSE(A = A, b = b, LambDa = LambDa)
Rlse_ans, Rlse_loss = Lse.LsE(input_x = input_x)


## Newton
Newton = NEWTON(A = A, b = b)
Newton_ans, Newton_loss = Newton.nEwToN(input_x = input_x)


## output
output_L_A = OUTPUT_line_ans(input_x = input_x, b = b)
output_L_A.print_ans(name = "LSE", parameter = Rlse_ans, loss = Rlse_loss)
output_L_A.print_ans(name = "Newton's Method", parameter = Newton_ans, loss = Newton_loss)

## plot 

LSE_title = "LSE: Bases = " + str(polynomial_Bases) + ", Lambda = " + str(LambDa)
Newton_title = "Newton: Bases = " + str(polynomial_Bases)

output_L_A.ploting(LSE_title = LSE_title, \
    Newton_title = Newton_title, \
    Lse_ans = Rlse_ans, \
    Newton_ans = Newton_ans )