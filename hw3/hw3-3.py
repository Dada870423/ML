from gaussian import *
import numpy as np
import copy


def poly_base(input_x, basis):
    tmp = np.zeros(basis)
    for iter_j in range(basis - 1, -1, -1):
        tmp[iter_j] = input_x ** iter_j

    return tmp

def poly_basis_generator(n, W, a):
    """
    W: (n, 1)
    """
    W = np.array(W).reshape(1, -1)
    x = np.random.uniform(low=-1.0, high=1.0)
    A = np.array([x ** i for i in range(n)])
    return x, W.dot(A)[0] + uni_gaussian_generator(0, a)

def uni_gaussian_generator(mean, variance):
    normal = np.random.uniform(size=12).sum() - 6
    return normal * np.sqrt(variance) + mean


b = int(input("b : "))
basis = int(input("basis : "))
a = int(input("a : "))
w = list()

for base in range(basis):
    get_w_str = "W " + str(base) + " : "
    w_input = int(input("W" + str(base) + " : "))
    w.append(w_input)

II = np.identity(basis)
#BI = II * b
print(w)

#prior_m = np.zeros(basis, dtype = float)
prior_m = np.zeros((basis, 1))
prior_S = np.identity(basis) * b
#prior_S = II * b



#mEaN = np.zeros(basis, dtype = float)
#A = np.zeros(0)
X = np.zeros(0)
Y = np.zeros(0)

## iter 1

#print(A)
#CovarianceMatrix = a * (A.T @ A) + BI
#inv_CovarianceMatrix_S = np.linalg.pinv(CovarianceMatrix)
#
#mEaN = a * (inv_CovarianceMatrix_S @ A.T @ Y)
#
#print(CovarianceMatrix)
#print(mEaN)

#mean=np.zeros((basis))


#inv_posterior_S = prior_S + (1 / a)  * A.T @ A
#posterior_S = np.linalg.inv(inv_posterior_S)
#
#posterior_m = (posterior_S @ prior_S @ prior_m) + (1 / a) * A.T * para_y
#
#prior_m = posterior_m
#prior_S = inv_posterior_S

## iter 2
cnt = 0
for i in range(1, 5000):
    cnt = cnt + 1
    #para_x, para_y = Poly_generator(basis = basis, a = a, w = w)
    para_x, para_y = poly_basis_generator(n = basis, W = w, a = a)
    print(para_x, para_y)
    #X = np.append(X, para_x)
    #Y = np.append(Y, para_y)
    #A = poly_base(input_x = para_x, basis = basis)
    X = np.array([[para_x ** j for j in range(args.n)]])
    #print(A)




    #inv_posterior_S = prior_S + (1 / a) * A.T @ A
    inv_posterior_S = prior_S + (1 / args.a) * X.T.dot(X)
    posterior_S = np.linalg.inv(inv_posterior_S)
    
    #posterior_m = posterior_S @ ((prior_S @ prior_m) + (1 / a) * A.T * para_y)
    #prior_SM = (prior_S @ prior_m)

    #a_1_AT_PARA_Y = (1 / a) * A.T * para_y

    #prior_A_1 = prior_SM + a_1_AT_PARA_Y

    #posterior_m = posterior_S @ prior_A_1

    posterior_m = posterior_S.dot(prior_S.dot(prior_m) + (1 / args.a) * X.T * y)




    predictive_m = X.dot(posterior_m)
    predictive_S = args.a + X.dot(posterior_S.dot(X.T))

    print('Add data point (%s, %s):\n' % (x, y))
    print('Posterior mean:\n', posterior_m, '\n')
    print('Posterior variance:\n', posterior_S, '\n')
    print('Predictive distribution ~ N(%s, %s)' % (predictive_m.item(), predictive_S.item()))
    print('-'*70)


    prior_m = posterior_m
    prior_S = inv_posterior_S






#    variance_new = (a * (A.T @ A)) + S
#    inv_CovarianceMatrix = np.linalg.inv(variance)
#    mean_new = inv_CovarianceMatrix @ (a * A.T @ Y + variance @ mean)
#
#    S = np.linalg.pinv(variance)

    #variance_new = copy.deepcopy(np.linalg.inv(a * (A.T @ A) + S))

    #print(np.shape(mean))

    #mean_new = copy.deepcopy(variance_new @ ((a * (A.T @ Y)) + (S @ mean)))


    print('Update times: ', cnt)
    if cnt > 1000 and abs(predictive_S.item() - a) < 1e-3:
        break
print('Update times_out: ', cnt)
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
