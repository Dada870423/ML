import numpy as np
from UTIL import *
import copy
from numba import jit


input_N = 60000
lamBda = np.ones(10)
probability = np.random.rand(28 * 28, 10)
hidden_W = np.ones((input_N, 10))
#Binomial_matrix = Binomial_matrix
jimmy = 100 # big number!!
Label_fptr = Get_label_fptr(label_file = "file/train-labels-idx1-ubyte")
label = np.zeros((10, 500))


@jit
def E_step(Binomial_matrix):
    for iter_image in range(input_N):
        for iter_digit in range(10):
            hidden_W[iter_image][iter_digit] = lamBda[iter_digit]
    for iter_image in range(input_N):
        for iter_pixel in range(28 * 28):
            for iter_digit in range(10):
            
                if Binomial_matrix[iter_image][iter_pixel] == 1:
                    hidden_W[iter_image][iter_digit] *= probability[iter_pixel][iter_digit]
                else:
                    hidden_W[iter_image][iter_digit] *= (1 - probability[iter_pixel][iter_digit])
                ## deal with underflow
            if iter_pixel % 10 == 0:
                hidden_W[iter_image] = norm_probability(hidden_W[iter_image])
                #hidden_W[iter_image][iter_digit] *= jimmy
        hidden_W[iter_image] = norm_probability(hidden_W[iter_image])
        
    hidden_W[hidden_W<0.001] = 0.001


@jit
def M_step(Binomial_matrix):
    for iter_digit in range(10):
        lamBda[iter_digit] = 0.0
        for iter_image in range(input_N):
            lamBda[iter_digit] += hidden_W[iter_image][iter_digit]
    for iter_digit in range(10):
        for iter_pixel in range(28 * 28):
            probability[iter_pixel][iter_digit] = 0.0
    
    for iter_pixel in range(28 * 28):
        for  iter_digit in range(10):
            for iter_image in range(input_N):
                if Binomial_matrix[iter_image][iter_pixel] == 1:
                    probability[iter_pixel][iter_digit] += (hidden_W[iter_image][iter_digit])
                    #print(hidden_W[iter_image][iter_digit], probability[iter_pixel][iter_digi
                        
            
            probability[iter_pixel][iter_digit] /= lamBda[iter_digit]
    
    lamBda= norm_probability(lamBda)


@jit
def Test():
    GroundTruth = np.zeros((10, 10))
    items = Get_label_100()
    for iter_digit in range(10):
        for iter_item in range(int(items[iter_digit])):
            ans = Cal_w(image_th = int(label[iter_digit][iter_item]))
            print("-- ", iter_digit, " --   : ", ans.argmax())
            GroundTruth[iter_digit][ans.argmax()] += 1
    return GroundTruth

@jit
def Cal_w(Binomial_matrix, image_th):
    ans = np.ones(10)
    for iter_pixel in range(28 * 28):
        for iter_digit in range(10):
        
            if Binomial_matrix[image_th][iter_pixel] == 1:
                ans[iter_digit] *= probability[iter_pixel][iter_digit]
            else:
                ans[iter_digit] *= (1 - probability[iter_pixel][iter_digit])
            ## deal with underflow
        if iter_pixel % 10 == 0:
            ans = norm_probability(ans)
            #hidden_W[iter_image][iter_digit] *= jimmy
    ans = norm_probability(ans)
    return ans

@jit
def Get_label_100():
    items = np.zeros(10)
    for iter_label in range(1000):
        label_now = get_label(Label_fptr)
        xxx = int(items[label_now])
        label[label_now][xxx] = iter_label
        items[label_now] = items[label_now] + 1
    for iter_digit in range(10):
        for iter_item in range(int(items[iter_digit])):
            print(label[iter_digit][iter_item], end = " ")
        print("--------  ", iter_digit, "\n")
    print("items", items)
    return items


