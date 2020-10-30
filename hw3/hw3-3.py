from gaussian import *
import numpy as np
import copy

def funCtion(x, parameter):
    lead = len(parameter)
    y = 0
    for iter_i in range(len(parameter) - 1, -1, -1):
        tmp = x ** iter_i
        #y = y + tmp * parameter[len(parameter) - iter_i - 1]
        y = y + tmp * parameter[iter_i]
    return y


def ploting(Ground_truth, Predict_result, Sample_X, Sample_Y, Incomes_m_10, Incomes_m_50):
    plot_x = np.arange(-1.5, 1.5, 0.01)
    
    ## Ground_truth
    Ground_truth_plot_y = funCtion(plot_x, Ground_truth)
    plt.subplot(221)
    plt.plot(plot_x, Ground_truth_plot_y, color = "black", label = "w")
    plt.title("Ground truth")
    plt.legend(loc = 'upper right')

    ## Predict_result
    Predict_result_plot_y = funCtion(plot_x, Predict_result)
    plt.subplot(222)
    plt.plot(plot_x, Predict_result_plot_y, color = "black", label = "Posterior mean")
    plt.scatter(Sample_X, Sample_Y, c = "aqua", label = "Sample", marker = "x")
    plt.title("Predict result")
    plt.legend(loc = 'upper right')
    
    ## After 10 incomes
    Incomes_10_plot_y = funCtion(plot_x, Incomes_m_10)
    plt.subplot(223)
    plt.plot(plot_x, Incomes_10_plot_y, color = "black", label = "Posterior mean")
    plt.scatter(Sample_X[:10], Sample_Y[:10], c = "aqua", label = "10 Samples", marker = "x")
    plt.title("After 10 incomes")
    plt.legend(loc = 'upper right')

    ## After 50 incomes
    Incomes_50_plot_y = funCtion(plot_x, Incomes_m_50)
    plt.subplot(224)
    plt.plot(plot_x, Incomes_50_plot_y, color = "black", label = "Posterior mean")
    plt.scatter(Sample_X[:50], Sample_Y[:50], c = "aqua", label = "50 Samples", marker = "x")
    plt.title("After 50 incomes")
    plt.legend(loc = 'upper right')
    plt.show()


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


cnt = 0
for iter_i in range(1, 5000):
    cnt = cnt + 1
    para_x, para_y = Poly_generator(basis = basis, a = a, w = w)
    print(para_x, para_y)
    X_set = np.append(X_set, para_x)
    Y_set = np.append(Y_set, para_y)
    X = np.array([[para_x ** base for base in range(basis)]])

    if iter_i == 9:
        incomes_m_10 = prior_m
        incomes_S_10 = prior_S
    elif iter_i == 49:
        incomes_m_50 = prior_m
        incomes_S_50 = prior_S




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


    prior_m = posterior_m
    prior_S = inv_posterior_S



    print("Update times: ", cnt)
    if cnt > 1000 and abs(predictive_S[0][0] - a) < 1e-3:
        break
print('Update times_out: ', cnt)

ploting(Ground_truth = w, Predict_result = posterior_m, Sample_X = X_set, Sample_Y = Y_set,\
        Incomes_m_10 = incomes_m_10, Incomes_m_50 = incomes_m_50)