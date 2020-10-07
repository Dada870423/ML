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





# def funCtion(x, parameter):
#     lead = len(parameter)
#     y = 0
#     for iter_i in range(len(parameter) - 1, -1, -1):
#         tmp = x ** iter_i
#         y = y + tmp * parameter[len(parameter) - iter_i - 1]
#     return y

filename = sys.argv[1]
polynomial_Bases = int(sys.argv[2])
LambDa = float(sys.argv[3])

## get matrix A & b
a_matrix = A_MATRIX(filename = filename, polynomial_Bases = polynomial_Bases)
A = a_matrix.getA()
b = a_matrix.getB()
input_x = a_matrix.getX()
#print(A)
# print("A:", A)

## LSE
Lse = LSE(A = A, b = b, LambDa = LambDa)
Rlse_ans, Rlse_loss = Lse.LsE(input_x = input_x)


## output

output_L_A = OUTPUT_line_ans(input_x = input_x, b = b)
output_L_A.print_ans(name = "LSE", parameter = Rlse_ans, loss = Rlse_loss)

## print("LSE:")
## print("Fitting line: ", end = " ")
## 
## 
## for iter_ in range(len(Rlse_ans)):
##     if iter_ == len(Rlse_ans) - 1:
##         print("( ", Rlse_ans[iter_], " )")
##     else:
##         print("( ", Rlse_ans[iter_], " ) X^", len(Rlse_ans) - iter_ - 1, " + ", end = "")
## #print("Rlse_ans:", Rlse_ans)
## print("Total error: ", Rlse_loss)




## Newton
Newton = NEWTON(A = A, b = b)
Newton_ans, Newton_loss = Newton.nEwToN(input_x = input_x)


output_L_A.print_ans(name = "Newton's Method", parameter = Newton_ans, loss = Newton_loss)


#  print("Newton's Method:")
#  print("Fitting line: ", end = " ")
#  
#  for iter_ in range(len(Newton_ans)):
#      if iter_ == len(Newton_ans) - 1:
#          print("( ", Newton_ans[iter_], " )")
#      else:
#          print("( ", Newton_ans[iter_], " ) X^", len(Newton_ans) - iter_ - 1, " + ", end = "")
#  #print("Newton_ans:", Newton_ans)
#  print("Total error: ", Newton_loss)

## plot Newton
## plot RLSE

LSE_title = "LSE: Bases = " + str(polynomial_Bases) + ", Lambda = " + str(LambDa)
Newton_title = "Newton: Bases = " + str(polynomial_Bases)

output_L_A.ploting(LSE_title = LSE_title, \
    Newton_title = Newton_title, \
    Lse_ans = Rlse_ans, \
    Newton_ans = Newton_ans )


##  plot_x = np.arange(min(input_x) - 1, max(input_x) + 1, 0.01)
##  RLSE_plot_y = funCtion(plot_x, Rlse_ans)
##  plt.subplot(121)
##  plt.plot(plot_x, RLSE_plot_y, color = 'aqua', label = "Rlse")
##  plt.scatter(input_x, b, c = "red", label = "TestData")
##  LSE_title = "RLSE: Bases = " + str(polynomial_Bases) + ", Lambda = " + str(LambDa)
##  plt.title(LSE_title)
##  plt.legend(loc = 'upper right')
##  #plt.show()
##  
##  
##  
##  
##  plot_x = np.arange(min(input_x) - 1, max(input_x) + 1, 0.01)
##  Newton_plot_y = funCtion(plot_x, Newton_ans)
##  plt.subplot(122)
##  plt.plot(plot_x, Newton_plot_y, color = 'aqua', label = "Newton")
##  plt.scatter(input_x, b, c = "red", label = "TestData")
##  Newton_title = "Newton: Bases = " + str(polynomial_Bases)
##  plt.title(Newton_title)
##  plt.legend(loc = 'upper right')
##  plt.show()

