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

    for i in range(100000):
        para_x, y = Poly_generator(basis = basis, a = a, w = w)
        samples.append(y)
    
    plt.hist(samples, 50)
    plt.title("basis:{},a:{}".format(2, 10))
else:
    print("Univariate gaussian")
    mean = int(input("mean : "))
    varinance = int(input("varinance : "))
    for i in range(100000):
       SSSS = Univariate_generator(mean = mean, varinance = varinance)
       samples.append(SSSS)

    plt.hist(samples, 50)
    plt.title("mean: " + str(mean) + " varinance: " + str(varinance))
plt.show()