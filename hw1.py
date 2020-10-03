import sys
import csv
import numpy as np  
import matplotlib.pyplot as plt 
import math
import os

from A_MATRIX import A_MATRIX
from RLSE import RLSE
def plot(x1,b,parameters_rlse):
    #rlse
    plt.title('rlse')
    plt.plot(x1,b,'ro')
    x1_min=min(x1)
    x1_max=max(x1)
    x=np.linspace(x1_min-1,x1_max+1,500)
    y=np.zeros(x.shape)
    for i in range(len(parameters_rlse)):
        y+=parameters_rlse[i]*np.power(x,i)
    plt.plot(x,y,'-k')
    plt.show()


filename = sys.argv[1]
polynomial_Bases = int(sys.argv[2])
LambDa = float(sys.argv[2])

## get matrix A & b
a_matrix = A_MATRIX(filename = filename, polynomial_Bases = polynomial_Bases)
A = a_matrix.getA()
b = a_matrix.getB()
#print(A)
# print("A:", A)

## RLSE
Rlse = RLSE(A = A, b = b, LambDa = LambDa)
Rlse_ans, Rlse_loss = Rlse.rLsE()

print("Rlse_ans", Rlse_ans)
print("Rlse_loss", Rlse_loss)

xx = a_matrix.getX()

xx1 = np.asarray(xx,dtype='float').reshape((-1,1))
b1 = np.asarray(b,dtype='float').reshape((-1,1))
Rlse_ans1 = np.asarray(Rlse_ans,dtype='float').reshape((-1,1))

plot(xx1, b1, Rlse_ans1)
#print(xxx - b)
#print(b)
#haha = abs((xxx - b))
#print("haha: ", haha.sum())
#ans = (xxx - b)*(xxx - b)
#print("ans: ", ans.sum())
#print(ans)
#print((xxx - b)*(xxx - b))

#

#plt.scatter(xx, b, color = 'aqua', label = "input")
#plt.plot(xx, xxx, color = 'brown', label = "line")
#plt.legend(loc = 'upper right')

#plt.show()


