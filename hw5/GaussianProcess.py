import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class GaussianProcess(object):
    def __init__(self, alpha, lengthscale, variance):
        self.beta = 5
        self.delta = 1
        self.alpha = alpha
        self.lengthscale = lengthscale
        self.variance = variance
        self.plot_x = np.arange(-60, 60, 0.1)


    def Cal_kernel(self, X1, X2):
        Kernel = np.zeros((len(X1), len(X2)))
        for iter_y in range(len(X1)):
            for iter_x in range(len(X2)):
                fucking = 1 + (((X1[iter_y] - X2[iter_x]) ** 2) / \
                    (2 * self.alpha * (self.lengthscale ** 2)))
                Kernel[iter_y][iter_x] = (self.variance ** 2) * (fucking ** (- self.alpha))
        return Kernel


    def opt_Cal_kernel(self, X1, X2, alpha , variance, lengthscale):
        Kernel = np.zeros((len(X1), len(X2)))
        for iter_y in range(len(X1)):
            for iter_x in range(len(X2)):
                fucking = 1 + (((X1[iter_y] - X2[iter_x]) ** 2) / \
                    (2 * alpha * (lengthscale ** 2)))
                Kernel[iter_y][iter_x] = (variance ** 2) * (fucking ** (- alpha))
        return Kernel


    def get_data(self, filename = "input.data"):
        list_x = list()
        list_y = list()
    
        fptr = open(filename)
    
        line = fptr.readline()
        while line:
            datapoint = line.split(" ")
            list_x.append(float(datapoint[0]))
            list_y.append(float(datapoint[1]))
            line = fptr.readline()
        fptr.close()
        
        self.x = np.asarray(list_x)
        self.y = np.asarray(list_y)
    
        return self.x, self.y


    def Cal_mean(self, x, y):
        sample = np.arange(-60, 60, 0.1)

        self.Kernel_xn_xm = self.Cal_kernel(X1 = x, X2 = x)
        self.C_xn_xm = self.Kernel_xn_xm + (1 / self.beta * self.delta)
        self.K_X_Xstar = self.Cal_kernel(X1 = x, X2 = self.plot_x)

        mean = self.K_X_Xstar.T @ (np.linalg.inv(self.C_xn_xm)) @ y
        return mean


    def Cal_var(self, x, y):
        K_Xstar_Xstar = np.zeros(len(self.plot_x))
        variance = np.zeros(len(self.plot_x))

        for iter_pixel in range(len(K_Xstar_Xstar)):
            fucking = 1 + (((self.plot_x[iter_pixel] - self.plot_x[iter_pixel]) ** 2) / \
                    (2 * self.alpha * (self.lengthscale ** 2)))
            K_Xstar_Xstar[iter_pixel] = 1 / self.beta + fucking
            variance[iter_pixel] = np.abs(K_Xstar_Xstar[iter_pixel] - self.K_X_Xstar[:,iter_pixel] @ (np.linalg.inv(self.C_xn_xm)) @ self.K_X_Xstar[:,iter_pixel])
        return variance


    def Log_Likelihood(self, x0):
        alpha, lengthscale, variance = x0[0], x0[1], x0[2]
        K = self.opt_Cal_kernel(X1 = self.x, X2 = self.x, alpha = alpha, lengthscale = lengthscale, variance = variance)
        lgggg = np.abs(np.linalg.det(K))
        if np.linalg.matrix_rank(K) != len(self.x) or lgggg == 0:
            return 1000
        
        lg_norm_K = np.log(lgggg)
        inv_K = np.linalg.inv(K)
        ans = 0.5 * (self.y.T @ inv_K @ self.y + lg_norm_K + len(self.x) * np.log(2 * np.pi))
        if ans < 0.001:
            return 1000
        print(ans)
        return ans



    def optimize(self):
        error = 1000
        inits = np.arange(1, 10, 1)
        opt_alpha, opt_lengthscale = 1.0, 1.0
        for iter_alpha in inits:
            for iter_lengthscale in inits:
                for iter_variance in inits:
                    result = minimize(self.Log_Likelihood, x0 = [iter_alpha, iter_lengthscale, iter_variance], method = "SLSQP")
                    if result.fun < error:
                        error = result.fun
                        opt_alpha, opt_lengthscale, opt_variance = result.x
        return opt_alpha, opt_lengthscale, opt_variance, error

    
def funCtion(x, variance):
    var = np.zeros(len(variance))
    for iter_element in range(len(variance)):
        if variance[iter_element] == 0:
            print("fucking 0")
        var[iter_element] = 2 * (variance[iter_element] ** 0.5)
    return var


def plotting(mean, variance, opt_mean, opt_variance, para, x, y):
    plot_x = np.arange(-60, 60, 0.1)
    plt.subplot(121)
    plt.title("Gaussian Process Regression")
    plt.plot(plot_x, mean, 'coral')
    var = funCtion(x = plot_x, variance = variance)
    plt.fill_between(plot_x, mean - var, mean + var, color='aqua')
    plt.scatter(x, y, c = "black")

    plt.subplot(122)
    plt.title("Gaussian Process Regression (Optimized)")
    plt.plot(plot_x, opt_mean, 'coral')
    opt_var = funCtion(x = plot_x, variance = opt_variance)
    plt.fill_between(plot_x, mean - opt_var, mean + opt_var, color='aqua')
    plt.scatter(x, y, c = "black")
    annotated = "alpha : " + para[0] + "\nlengthscale : " + para[1] + "\nvariance : " + para[2]
    plt.text(0.5, -6.1, annotated, size=10, rotation=30.,ha="center", va="bottom",bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))

    plt.show() 
    
def print_Kernel(Kernel):
    row, column = Kernel.shape
    for iter_y in range(column):
        for iter_x in range(row):
            print(Kernel[iter_y][iter_x], end = " ")
        print()


## result
## 8.491449330946992 -2.4761190591796756 1.37230728866543


## reference
## https://peterroelants.github.io/posts/gaussian-process-kernels/