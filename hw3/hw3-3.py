from gaussian import *
from UTIL import *
import numpy as np
import copy




b = int(input("b : "))
basis = int(input("basis : "))
a = int(input("a : "))
w = list()

for base in range(basis):
    get_w_str = "W " + str(base) + " : "
    w_input = int(input("W" + str(base) + " : "))
    w.append(w_input)


prior_m = np.zeros((basis, 1))
prior_S = np.identity(basis) * b

X_set = np.zeros(0)
Y_set = np.zeros(0)


for iter_i in range(100000):
    para_x, para_y = Poly_generator(basis = basis, a = a, w = w)
    print(para_x, para_y)
    X_set = np.append(X_set, para_x)
    Y_set = np.append(Y_set, para_y)
    X = np.array([[para_x ** base for base in range(basis)]])

    inv_posterior_S = prior_S + (1 / a) * X.T @ X

    posterior_S = np.linalg.inv(inv_posterior_S)
    posterior_m = posterior_S @ ((prior_S @ prior_m) + (1 / a) * X.T * para_y)

    predictive_m = X @ posterior_m
    predictive_S = a + (X @ (posterior_S @ X.T))

    print("Add data point (", para_x, ", ", para_y, "):\n")
    print("Posterior mean:\n", posterior_m, "\n")
    print("Posterior variance:\n", posterior_S, "\n")
    print("Predictive distribution ~ N(", predictive_m[0][0], ", ", predictive_S[0][0], ")")
    print("-"*73)

    if iter_i == 9:
        incomes_m_10 = copy.deepcopy(posterior_m)
        incomes_S_10 = copy.deepcopy(posterior_S)
    if iter_i == 49:
        incomes_m_50 = copy.deepcopy(posterior_m)
        incomes_S_50 = copy.deepcopy(posterior_S)


    prior_m = posterior_m
    prior_S = inv_posterior_S



    print("Update times: ", iter_i)
    if iter_i > 51 and abs(predictive_S[0][0] - a) < 0.001:
        break

ploting(a = a, basis = basis, Ground_truth = w, Predict_result = posterior_m, Predict_result_S = posterior_S, \
        Sample_X = X_set, Sample_Y = Y_set,\
        Incomes_m_10 = incomes_m_10, Incomes_S_10 = incomes_S_10, \
        Incomes_m_50 = incomes_m_50, Incomes_S_50 = incomes_S_50)
