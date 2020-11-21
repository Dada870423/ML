import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class GaussianProcess(object):
    def __init__(self, beta = 5, delta = 1):
        self.beta = beta
        self.delta = delta
        self.plot_x = np.arange(-60, 60, 0.1)
    


    def Cal_kernel(self, X1, X2, alpha = 1, variance = 1, lengthscale = 1):
        Kernel = np.zeros((len(X1), len(X2)))
        for iter_y in range(len(X1)):
            for iter_x in range(len(X2)):
                fucking = 1 + (((X1[iter_y] - X2[iter_x]) ** 2) / \
                    (2 * alpha * (lengthscale ** 2)))
                Kernel[iter_y][iter_x] = (variance ** 2) * (fucking ** (- alpha))
        return Kernel
    
    #def Cal_small_kernel(self, X1, X2, alpha = 1, variance = 1, lengthscale = 1):
    #    fucking = 1 + (((X1[iter_y] - X2) ** 2) / (2 * alpha * (lengthscale ** 2)))
    #    return (variance ** 2) * (fucking ** (- alpha))
    
    
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
        ## cal K_X_Xstar
        self.Kernel_xn_xm = self.Cal_kernel(X1 = x, X2 = x)
        self.C_xn_xm = self.Kernel_xn_xm + (1 / self.beta * self.delta)
        self.K_X_Xstar = self.Cal_kernel(X1 = x, X2 = self.plot_x)
        #K_Xstar_Xstar = self.Cal_kernel(X1 = sample, X2 = sample) + 1 / self.beta

        mean = self.K_X_Xstar.T @ (np.linalg.inv(self.C_xn_xm)) @ y
        return mean




    def Cal_var(self, x, y):
        K_Xstar_Xstar = np.zeros(len(self.plot_x))
        variance = np.zeros(len(self.plot_x))
        print("variance", variance.shape)
        print("K_Xstar_Xstar", K_Xstar_Xstar.shape)
        print("K_X_Xstar", self.K_X_Xstar.shape)
        print("self.C_xn_xm", self.C_xn_xm.shape)

        print("test self.K_X_Xstar[:,iter_pixel]", self.K_X_Xstar[:,1].shape)


        for iter_pixel in range(len(K_Xstar_Xstar)):
            K_Xstar_Xstar[iter_pixel] = 1 / self.beta + 1
            variance[iter_pixel] = K_Xstar_Xstar[iter_pixel] - self.K_X_Xstar[:,iter_pixel] @ (np.linalg.inv(self.C_xn_xm)) @ self.K_X_Xstar[:,iter_pixel]
        return variance





    def Log_Likelihood(self, x0):
        #self.Kernel_xn_xm
        alpha, lengthscale = x0[0], x0[1]
        K = self.Cal_kernel(X1 = self.x, X2 = self.x, alpha = alpha, lengthscale = lengthscale)
        #K = self.Kernel_xn_xm
        lg_norm_K = np.log(np.abs(np.linalg.det(K)))
        inv_K = np.linalg.inv(K)
        ans = - 0.5 * (self.y.T @ inv_K @ self.y + lg_norm_K + len(self.x) * np.log(2 * np.pi))
        return ans



    def optimize(self):
        error = 1000
        inits = np.arange(1, 33, 3)
        opt_alpha, opt_lengthscale = 1.0, 1.0
        for iter_alpha in inits:
            for iter_lengthscale in inits:
                result = minimize(self.Log_Likelihood, x0 = [iter_alpha, iter_lengthscale])
                if result.fun < error:
                    error = result.fun
                    opt_alpha, opt_lengthscale = result.x
        return opt_alpha, opt_lengthscale






    def plotting(self, mean, variance, x, y):
        plt.title("Gaussian Process Regression")
        
        plt.plot(self.plot_x, mean, 'coral')
        vvvvvvar = self.funCtion(x = self.plot_x, variance = variance)
        #vvvar = 2*(variance**0.5)
        #plt.plot(self.plot_x, mean - vvvvvvar, color='coral')
        #plt.plot(self.plot_x, mean + vvvvvvar, color='coral')
        plt.fill_between(self.plot_x, mean - vvvvvvar, mean + vvvvvvar, color='aqua')
        plt.scatter(x, y, c = "black")

        plt.show()
    
    
    def funCtion(self, x, variance):
        var = np.zeros(len(variance))
        for iter_element in range(len(variance)):
            if variance[iter_element] == 0:
                print("fucking 0")
            var[iter_element] = 2 * (variance[iter_element] ** 0.5)
        return var
    
    
    def print_Kernel(self, Kernel):
        row, column = Kernel.shape
        for iter_y in range(column):
            for iter_x in range(row):
                print(Kernel[iter_y][iter_x], end = " ")
            print()









## reference
## https://peterroelants.github.io/posts/gaussian-process-kernels/