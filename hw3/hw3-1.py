from gaussian import *

samples = list()

mode = input("Univariate gaussian, Polynomial basis linear model : ")

if "p" in mode or "P" in mode:
    print("Polynomial basis linear model")
    basis = int(input("basis : "))
    a = int(input("a : "))
    w = list()
    for base in range(basis):
        get_w_str = "W " + str(base) + " : "
        w_input = int(input("W" + str(base) + " : "))
        w.append(w_input)

    X_samples = list()
    Y_samples = list()
    for i in range(100000):
        para_x, para_y = Poly_generator(basis = basis, a = a, w = w)
        X_samples.append(para_x)
        Y_samples.append(para_y)
    

    plt.scatter(X_samples, Y_samples, c = "aqua", label = "Sample", marker = "x")
    plt.title("basis: " + str(basis) + " , a: " + str(a) + " , w: " + str(w))
else:
    print("Univariate gaussian")
    mean = int(input("mean : "))
    varinance = int(input("varinance : "))
    for i in range(100000):
       SSSS = Univariate_generator(mean = mean, variance = varinance)
       samples.append(SSSS)

    plt.hist(samples, 50)
    plt.title("mean: " + str(mean) + " varinance: " + str(varinance))
plt.show()