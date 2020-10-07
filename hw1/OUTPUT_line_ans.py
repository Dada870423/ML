import numpy as np  
import matplotlib.pyplot as plt

class OUTPUT_line_ans(object):
    def __init__(self, input_x, b):
        self.input_x = input_x
        self.b = b

    def funCtion(self, x, parameter):
        lead = len(parameter)
        y = 0
        for iter_i in range(len(parameter) - 1, -1, -1):
            tmp = x ** iter_i
            y = y + tmp * parameter[len(parameter) - iter_i - 1]
        return y

    def print_ans(self, name, parameter, loss):
        print(name, ":")
        print("Fitting line: ", end = " ")
        for iter_i in range(len(parameter)):
            if iter_i == len(parameter) - 1:
                print("( ", parameter[iter_i], " )")
            else:
                print("( ", parameter[iter_i], " ) X^", len(parameter) - iter_i - 1, " + ", end = "")
        print("Total error: ", loss)

    def ploting(self, LSE_title, Newton_title, Lse_ans, Newton_ans):
        plot_x = np.arange(min(self.input_x) - 1, max(self.input_x) + 1, 0.01)
        
        ## LSE
        LSE_plot_y = self.funCtion(plot_x, Lse_ans)
        plt.subplot(121)
        plt.plot(plot_x, LSE_plot_y, color = 'aqua', label = "LSE")
        plt.scatter(self.input_x, self.b, c = "red", label = "TestData")
        plt.title(LSE_title)
        plt.legend(loc = 'upper right')

        ## Newton
        Newton_plot_y = self.funCtion(plot_x, Newton_ans)
        plt.subplot(122)
        plt.plot(plot_x, Newton_plot_y, color = 'aqua', label = "Newton")
        plt.scatter(self.input_x, self.b, c = "red", label = "TestData")
        plt.title(Newton_title)
        plt.legend(loc = 'upper right')
        plt.show()





