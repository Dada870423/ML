import numpy as np
import argparse
from GaussianProcess import *

varss = [0.001, 1.37230728866543, 2.5, 3.5]


GP0 = GaussianProcess(alpha = 8.491449330946992, lengthscale = 2.4761190591796756, variance = varss[0])

x, y = GP0.get_data()
mean0 = GP0.Cal_mean(x = x, y = y)
variance0 = GP0.Cal_var(x = x, y = y)

GP1 = GaussianProcess(alpha = 8.491449330946992, lengthscale = 2.4761190591796756, variance = varss[1])
mean1 = GP1.Cal_mean(x = x, y = y)
variance1 = GP1.Cal_var(x = x, y = y)

GP2 = GaussianProcess(alpha = 8.491449330946992, lengthscale = 2.4761190591796756, variance = varss[2])
mean2 = GP2.Cal_mean(x = x, y = y)
variance2 = GP2.Cal_var(x = x, y = y)

GP3 = GaussianProcess(alpha = 8.491449330946992, lengthscale = 2.4761190591796756, variance = varss[3])
mean3 = GP3.Cal_mean(x = x, y = y)
variance3 = GP3.Cal_var(x = x, y = y)

plotting4var(mean0, variance0, mean1, variance1, mean2, variance2, mean3, variance3, varss, x, y)