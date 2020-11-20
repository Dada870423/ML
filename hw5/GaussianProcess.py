import numpy as np



def Cal_kernel(X1, X2, alpha = 1, variance = 1, lengthscale = 1):
    Kernel = np.zeros((len(X1), len(X2)))
    for iter_y in range(len(X1)):
        for iter_x in range(len(X2)):
            fucking = 1 + (((X1[iter_y] - X2[iter_x]) ** 2) / \
                (2 * alpha * (lengthscale ** 2)))
            Kernel[iter_y][iter_x] = (variance ** 2) * (fucking ** (- alpha))
    return Kernel




def get_data(filename = "input.data"):
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
    




def print_Kernel(Kernel):
    row, column = Kernel.shape
    for iter_y in range(column):
        for iter_x in range(row):
            print(Kernel[iter_y][iter_x], end = " ")
        print()









## reference
## https://peterroelants.github.io/posts/gaussian-process-kernels/