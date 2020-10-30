import numpy as np
import math
import random
import matplotlib.pyplot as plt

def Univariate_generator(mean, variance):
    S = 100
    while S >= 1 or S == 0:
    	U = random.uniform(-1, 1)
    	V = random.uniform(-1, 1)
    	S = U **2 + V **2
    z0 = math.sqrt(-2 * math.log(S)) * U / math.sqrt(S)
    
    point = z0 * (variance ** (0.5)) + mean

    return point


def Poly_generator(basis, a, w):
    E = Univariate_generator(mean = 0, variance = a)
    para_x = random.uniform(-1, 1)
    x_vector = [math.pow(para_x, base) for base in range(basis)]
    poly_sum = 0
    for base in range(basis):
    	poly_sum = poly_sum + x_vector[base] * w[base]

    return para_x, poly_sum + E