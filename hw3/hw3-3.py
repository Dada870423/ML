from gaussian import *
import numpy as np
import copy


def poly_base(input_x, basis, A):
    tmp = np.zeros(basis)
    for iter_j in range(basis - 1, -1, -1):
        tmp[iter_j] = input_x[-1] ** iter_j

    return np.append(A, tmp).reshape(len(input_x), basis)






b = int(input("b : "))
basis = int(input("basis : "))
a = int(input("a : "))
w = list()

for base in range(basis):
    get_w_str = "W " + str(base) + " : "
    w_input = int(input("W" + str(base) + " : "))
    w.append(w_input)

II = np.identity(basis)
BI = II * b

S = copy.deepcopy(BI)

inv_S = np.linalg.inv(BI)

variance = copy.deepcopy(inv_S)


variance=(1/b)*np.identity(basis)

#mEaN = np.zeros(basis, dtype = float)
A = np.zeros(0)
X = np.zeros(0)
Y = np.zeros(0)

## iter 1
para_x, para_y = Poly_generator(basis = basis, a = a, w = w)
X = np.append(X, para_x)
Y = np.append(Y, para_y)

A = poly_base(input_x = X, basis = basis, A = A)
#print(A)
#CovarianceMatrix = a * (A.T @ A) + BI
#inv_CovarianceMatrix_S = np.linalg.pinv(CovarianceMatrix)
#
#mEaN = a * (inv_CovarianceMatrix_S @ A.T @ Y)
#
#print(CovarianceMatrix)
#print(mEaN)

#mean=np.zeros((basis))


tri = a * (A.T @ A) + S

inv_tri = np.linalg.inv(tri)

mean = a * (inv_tri @ A.T @ Y)
variance = inv_tri

M_pron = copy.deepcopy(mean)

S_pron = tri



#variance = np.linalg.inv(a * (A.T @ A) + BI)
#S = np.linalg.pinv(variance)
#print(np.shape(variance_new))
#print(np.shape(X.T))

#mean = a * (variance @ A.T @ Y)

print(X, Y)
print(A)
print(S)
print(variance)
print(mean)


## iter 2
for i in range(1, 1000):
    para_x, para_y = Poly_generator(basis = basis, a = a, w = w)
    print(para_x, para_y)
    X = np.append(X, para_x)
    Y = np.append(Y, para_y)
    A = poly_base(input_x = X, basis = basis, A = A)
    if i == 8:
        print(A)
        print(X, Y)



    tir = a * (A.T @ A) + S_pron
    inv_tri = np.linalg.inv(tri)

    mean_pre = inv_tri @ (a * A.T @ Y + S_pron @ M_pron)
    variance = A.T @ inv_tri @ A

    mean = mean_pre.T @ A.T



    M_pron = copy.deepcopy(mean)
    S_pron = np.linalg.inv(variance)





#    variance_new = (a * (A.T @ A)) + S
#    inv_CovarianceMatrix = np.linalg.inv(variance)
#    mean_new = inv_CovarianceMatrix @ (a * A.T @ Y + variance @ mean)
#
#    S = np.linalg.pinv(variance)

    #variance_new = copy.deepcopy(np.linalg.inv(a * (A.T @ A) + S))

    #print(np.shape(mean))

    #mean_new = copy.deepcopy(variance_new @ ((a * (A.T @ Y)) + (S @ mean)))

    print('Posterior mean:')
    print(mean)
    print()
    print('Posterior variance:')
    print(variance)
    print()

    #mean = copy.deepcopy(mean_new)
    #variance = copy.deepcopy(variance_new)

#    CovarianceMatrix = a * (A.T @ A) + inv_CovarianceMatrix_S
#    inv_CovarianceMatrix = np.linalg.pinv(CovarianceMatrix)
#
#    mEaN = inv_CovarianceMatrix @ (a * A.T @ Y + inv_CovarianceMatrix_S @ mEaN)
#    inv_CovarianceMatrix_S = inv_CovarianceMatrix
#    print("#####################\n")
#
#
#    print(CovarianceMatrix, "\n\n")
#    print(mEaN)

    #print("mean: ", np.mean(X_Y), "var: ", np.cov(X_Y))
