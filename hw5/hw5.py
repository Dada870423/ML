import numpy as np
import argparse
from GaussianProcess import *


parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type = float, default =  1.0, help = "alpha")
parser.add_argument("--lengthscale", type = float, default =  1.0, help = "lengthscale")
parser.add_argument("--variance", type = float, default = 1.0, help = "variance")
input_ = parser.parse_args()

GP = GaussianProcess(alpha = input_.alpha, lengthscale = input_.lengthscale, variance = input_.variance)
x, y = GP.get_data()

mean = GP.Cal_mean(x = x, y = y)
variance = GP.Cal_var(x = x, y = y)


opt_alpha, opt_lengthscale, opt_variance, error = GP.optimize()

opt_GP = GaussianProcess(alpha = opt_alpha, lengthscale = opt_lengthscale, variance = opt_variance)

opt_mean = opt_GP.Cal_mean(x = x, y = y)
opt_variance = opt_GP.Cal_var(x = x, y = y)


para = [str(opt_alpha), str(opt_lengthscale), str(opt_variance)]
plotting(mean, variance, opt_mean, opt_variance, para, x, y)