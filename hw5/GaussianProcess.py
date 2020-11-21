import numpy as np
import matplotlib.pyplot as plt

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
    
    def Cal_small_kernel(self, X1, X2, alpha = 1, variance = 1, lengthscale = 1):
        Kernel = np.zeros(len(X1))
        for iter_y in range(len(X1)):
            fucking = 1 + (((X1[iter_y] - X2) ** 2) / \
                (2 * alpha * (lengthscale ** 2)))
            Kernel[iter_y] = (variance ** 2) * (fucking ** (- alpha))
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
        
        x = np.asarray(list_x)
        y = np.asarray(list_y)
    
        return x, y
        
    
    
    def Cal_mean(self, x, y):
        sample = np.arange(-60, 60, 0.1)
        ## cal K_X_Xstar
        Kernel_xn_xm = self.Cal_kernel(X1 = x, X2 = x)
        self.C_xn_xm = Kernel_xn_xm + (1 / self.beta * self.delta)
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





    def plotting(self, mean, variance, x, y):
        plt.title("Gaussian Process Regression")
        plt.scatter(x, y, c = "aqua")
        plt.plot(self.plot_x, mean, 'r')
        vvvvvvar = self.funCtion(x = self.plot_x, variance = variance)
        #vvvar = 2*(variance**0.5)
        plt.plot(self.plot_x, mean - vvvvvvar, color='pink')
        plt.plot(self.plot_x, mean + vvvvvvar, color='pink')

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