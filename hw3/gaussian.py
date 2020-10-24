import numpy as np
import math
import random
import matplotlib.pyplot as plt

def Univariate_generator(mean, varinance):
    #U = random.uniform(-1, 1)
    S = 100
    while S >= 1 or S == 0:
    	U = random.uniform(-1, 1)
    	V = random.uniform(-1, 1)
    	S = U **2 + V **2
    	#if S >= 1 or S == 0:
    		#print(S)
    z0 = math.sqrt(-2 * math.log(S)) * U / math.sqrt(S)
    SSSample = z0 * (varinance ** (0.5)) + mean
    return SSSample
    #return U, V, S


def Poly_generator(basis, a, w):
	E = Univariate_generator(mean = 0, varinance = a)
	para_x = random.uniform(-1, 1)
	x_vector = [math.pow(para_x, base) for base in range(basis)]
	poly_functino = 0
	for base in range(basis):
		poly_functino = x_vector[base] * w[base]

	#x_vector * w
	y = np.sum(poly_functino) + E

	return para_x, y





#print(Poly_generator(basis = 2, a = 10, w = [2, 5]))
#samples = list()
#for i in range(100000):
#	para_x, y = Poly_generator(basis = 2, a = 10, w = [2, 5])
#	samples.append(y)
#	#print("sample: ", SSSS)
#
#plt.hist(samples, 50)
#plt.title("basis:{},a:{}".format(2, 10))
#plt.show()


#samples = list()
#for i in range(100000):
#	SSSS = Univariate_generator(20, 10)
#	samples.append(SSSS)
#	#print("sample: ", SSSS)
#
#plt.hist(samples, 50)
#plt.title('mean:{},varinance:{}'.format(20, 10))
#plt.show()