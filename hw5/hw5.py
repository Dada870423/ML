import numpy as np
from GaussianProcess import *




GP = GaussianProcess(beta = 5, delta = 1)


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

opt_alpha, opt_lengthscale = GP.optimize()


print(opt_alpha, opt_lengthscale)