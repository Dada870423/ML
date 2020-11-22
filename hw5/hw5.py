import numpy as np
import argparse
from GaussianProcess import *


parser = argparse.ArgumentParser()

parser.add_argument("--alpha", type = float, default =  1.0, help = "alpha")
parser.add_argument("--lengthscale", type = float, default =  1.0, help = "lengthscale")
parser.add_argument("--variance", type = float, default = 1.0, help = "variance")

input_ = parser.parse_args()

print("para", input_.alpha, input_.lengthscale, input_.variance)

GP = GaussianProcess(alpha = input_.alpha, lengthscale = input_.lengthscale, variance = input_.variance)


x, y = GP.get_data()



#print_Kernel(Kernel)

#sample = np.arange(-60, 60, 0.01)
#
#mean = list()
#
#var = list()
#
#K_X_Xstar = Cal_kernel(X1 = x, X2 = x)


#test = GP.Cal_kernel(x, x)

#GP.print_Kernel(Kernel = test)

mean = GP.Cal_mean(x = x, y = y)
variance = GP.Cal_var(x = x, y = y)


#print(GP.Cal_kernel(4, 3))

GP.plotting(mean, variance, x, y)

print(x, y)

#opt_alpha, opt_lengthscale, opt_variance, error = GP.optimize()
#opt_alpha, opt_lengthscale = GP.optimize()

#print(opt_alpha, opt_lengthscale)

#print(opt_alpha, opt_lengthscale, opt_variance, error)

