import numpy as np
from GaussianProcess import *

x, y = get_data()

Kernel = Cal_kernel(X1 = x, X2 = x)

print_Kernel(Kernel)