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