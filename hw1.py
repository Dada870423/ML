import sys
import csv
import numpy as np  
import matplotlib.pyplot as plt 
import math
import os

from A_MATRIX import A_MATRIX

filename = sys.argv[1]
polynomial_Bases = int(sys.argv[2])
LambDa = float(sys.argv[2])

a_matrix = A_MATRIX(filename = filename, polynomial_Bases = polynomial_Bases)
A = a_matrix.getA()