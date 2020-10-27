from gaussian import *

samples_x = list()
samples_y = list()

b = int(input("b : "))
basis = int(input("basis : "))
a = int(input("a : "))
w = list()

for base in range(basis):
    get_w_str = "W " + str(base) + " : "
    w_input = int(input("W" + str(base) + " : "))
    w.append(w_input)

II = np.identity(basis)
var = II / b

print(var)


for i in range(10):
    para_x, para_y = Poly_generator(basis = basis, a = a, w = w)
    samples_x.append(para_x)
    samples_y.append(para_y)
