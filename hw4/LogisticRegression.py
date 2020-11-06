import numpy as np
import random
from gaussian import *
import copy

def generage_point(data_para_1, data_para_2, points_num):
    A = list()
    y = list()
    sample_point = list()
    ## class 1
    for iter_point in range(points_num):
        point = [Univariate_generator(mean = data_para_1[0], variance = data_para_1[1]), \
                Univariate_generator(mean = data_para_1[2], variance = data_para_1[3])]

        A.append(point + [1])
        sample_point.append(point)
        y.append(0.0)

    ## class 2
    for iter_point in range(points_num):
        point = [Univariate_generator(mean = data_para_2[0], variance = data_para_2[1]), \
                Univariate_generator(mean = data_para_2[2], variance = data_para_2[3])]

        A.append(point + [1])
        sample_point.append(point)
        y.append(1.0)

    return np.array(sample_point), np.array(A), np.array(y)


def Newton(w, A, y, lr = 0.01):
    #print(lr)
    pre_norm = 0
    norm = 100
    it = 0
    while np.linalg.norm(norm - pre_norm) >= 0.0001 or it < 3:
        it = it + 1
        Hessian, rank = get_Hessian(w = w, A = A)

        if rank == 3:
            inv_hessian = np.linalg.inv(Hessian)
            new_w = w + lr * (inv_hessian @ cal_J(w = w, A = A, y = y))
            
            pre_norm = copy.deepcopy(norm)
            norm = np.linalg.norm(inv_hessian)

            w = copy.deepcopy(new_w)
        else:
            print("do grandient desent")
            w = Gradient_desent(w = w, A = A, y = y, lr = 0.001)
            return w
    return w


def get_Hessian(w, A):
    m, trash_ = np.shape(A)
    D = np.zeros((m, m))
    for iter_diagonal in range(m):
        fucking = np.exp(- (A[iter_diagonal] @ w))
        D[iter_diagonal][iter_diagonal] = fucking / ((1 + fucking) ** 2)
    Hessian = A.T @ D @ A
    rank = np.linalg.matrix_rank(Hessian)
    return Hessian, rank




def Gradient_desent(w, A, y, lr = 0.01):
    norm = 100
    while norm >= 0.01:
        J = cal_J(w = w, A = A, y = y)
        new_w = w + (lr * J)
        w = copy.deepcopy(new_w)
        norm = np.linalg.norm(J)

    return w


def cal_J(w, A, y):
    fucking = 1 / (1 + np.exp(- (A @ w)))
    return A.T @ (y - fucking)


def predict(w, A, sample_point, points_num):
    y_pred = 1 / (1 + np.exp(- (A @ w)))
    c0 = np.array(sample_point[(y_pred < 0.5)])
    c1 = np.array(sample_point[(y_pred >= 0.5)])

    cluster_correct_1 = np.sum(y_pred[:50] < 0.5)
    cluster_correct_2 = np.sum(y_pred[50:] >= 0.5)

    print(w[0], "\n", w[1], "\n", w[2], "\n")
    print("Confusion Matrix:\n             Predict cluster 1   Predict cluster 2")
    print("Is cluster 1       ", cluster_correct_1, "                  ", points_num - cluster_correct_1)
    print("Is cluster 2       ", points_num - cluster_correct_2, "                  ", cluster_correct_2, "\n")
    print("Sensitivity (Successfully predict cluster 1): ", cluster_correct_1 / points_num)
    print("Specificity (Successfully predict cluster 2): ", cluster_correct_2 / points_num)

    return c0, c1

def ploting(sample_point, gradient_c0, gradient_c1, Newton_c0, Newton_c1, points_num):
    
    ## Ground truth
    plt.subplot(131)
    plt.scatter(sample_point[:points_num,0], sample_point[:points_num,1], c = "aqua", label = "Sample1", marker = "x")
    plt.scatter(sample_point[points_num:,0], sample_point[points_num:,1], c = "red", label = "Sample2", marker = "o")    
    plt.title("Ground_truth")
    plt.legend(loc = 'upper right')

    ## Gradient descent
    plt.subplot(132)
    plt.scatter(gradient_c0[:,0], gradient_c0[:,1], c = "aqua", label = "Sample1", marker = "x")
    plt.scatter(gradient_c1[:,0], gradient_c1[:,1], c = "red", label = "Sample2", marker = "o")
    plt.title("Gradient descent")
    plt.legend(loc = 'upper right')

    ## Newton's method
    plt.subplot(133)
    plt.scatter(Newton_c0[:,0], Newton_c0[:,1], c = "aqua", label = "Sample1", marker = "x")
    plt.scatter(Newton_c1[:,0], Newton_c1[:,1], c = "red", label = "Sample2", marker = "o")
    plt.title("Newton's method")
    plt.legend(loc = 'upper right')

    plt.show()