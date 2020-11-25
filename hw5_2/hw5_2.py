import numpy as np

from SVM import *

mode = input("your mode")

if mode == 0:
	print("comparison")
elif mode == 1:
	print("C-SVC")
elif mode == 2:
	print("user-defined kernel")
else:
	print("input the valid mode")