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

def Cal_var(a, basis, x, posterior_S):
    tmp = np.zeros(((len(x))))
    for iter_i in range(len(x)):
        XX = np.array([[x[iter_i] ** base for base in range(basis)]])
        tmp[iter_i] = a + (XX @ (posterior_S @ XX.T))
    return tmp




def ploting(a, basis, Ground_truth, Predict_result, Predict_result_S, Sample_X, Sample_Y, \
            Incomes_m_10, Incomes_S_10, Incomes_S_50, Incomes_m_50):
    plot_x = np.arange(-1.5, 1.5, 0.01)
    
    ## Ground_truth
    plt.subplot(221)
    Ground_truth_plot_y = funCtion(plot_x, Ground_truth)
    #Ground_truth_S = Cal_var(a = a, basis = basis, x = plot_x, posterior_S = np.identity(basis) / b)
    plt.plot(plot_x, Ground_truth_plot_y + a, color = "coral", label = "Variance")
    plt.plot(plot_x, Ground_truth_plot_y - a, color = "coral")
    plt.plot(plot_x, Ground_truth_plot_y, color = "black", label = "w")
    plt.title("Ground truth")
    plt.legend(loc = 'upper right')

    ## Predict_result
    plt.subplot(222)
    Predict_result_plot_y = funCtion(plot_x, Predict_result)
    Predict_result_S = Cal_var(a = a, basis = basis, x = plot_x, posterior_S = Predict_result_S)
    plt.plot(plot_x, Predict_result_plot_y, color = "black", label = "Posterior mean")
    plt.plot(plot_x, Predict_result_plot_y + Predict_result_S, color = "coral", label = "Variance")
    plt.plot(plot_x, Predict_result_plot_y - Predict_result_S, color = "coral")
    plt.scatter(Sample_X, Sample_Y, c = "aqua", label = "Sample", marker = "x")
    plt.title("Predict result")
    plt.legend(loc = 'upper right')
    
    ## After 10 incomes
    plt.subplot(223)
    Incomes_10_plot_y = funCtion(plot_x, Incomes_m_10)
    Incomes_10_S = Cal_var(a = a, basis = basis, x = plot_x, posterior_S = Incomes_S_10)
    plt.plot(plot_x, Incomes_10_plot_y, color = "black", label = "Posterior mean")
    plt.plot(plot_x, Incomes_10_plot_y + Incomes_10_S, color = "coral", label = "Variance")
    plt.plot(plot_x, Incomes_10_plot_y - Incomes_10_S, color = "coral")
    plt.scatter(Sample_X[:10], Sample_Y[:10], c = "aqua", label = "10 Samples", marker = "x")
    plt.title("After 10 incomes")
    plt.legend(loc = 'upper right')

    ## After 50 incomes
    plt.subplot(224)
    Incomes_50_plot_y = funCtion(plot_x, Incomes_m_50)
    Incomes_50_S = Cal_var(a = a, basis = basis, x = plot_x, posterior_S = Incomes_S_50)
    plt.plot(plot_x, Incomes_50_plot_y, color = "black", label = "Posterior mean")
    plt.plot(plot_x, Incomes_50_plot_y + Incomes_50_S, color = "coral", label = "Variance")
    plt.plot(plot_x, Incomes_50_plot_y - Incomes_50_S, color = "coral")
    plt.scatter(Sample_X[:50], Sample_Y[:50], c = "aqua", label = "50 Samples", marker = "x")
    plt.title("After 50 incomes")
    plt.legend(loc = 'upper right')
    plt.show()
    print(Incomes_m_10)
    print(Incomes_m_50)


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
