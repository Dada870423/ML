import numpy as np
import argparse
import random
from LogisticRegression import *

parser = argparse.ArgumentParser()
parser.add_argument("--N", type = int, default = 50)

parser.add_argument("--mx1", type = float, default =  1.0, help = "mx1")
parser.add_argument("--my1", type = float, default =  1.0, help = "my1")
parser.add_argument("--mx2", type = float, default = 10.0, help = "mx2")
parser.add_argument("--my2", type = float, default = 10.0, help = "my2")

parser.add_argument("--vx1", type = float, default =  2.0, help = "vx1")
parser.add_argument("--vy1", type = float, default =  2.0, help = "vy1")
parser.add_argument("--vx2", type = float, default =  2.0, help = "vx2")
parser.add_argument("--vy2", type = float, default =  2.0, help = "vy2")

input_ = parser.parse_args()

data_para_1 = [input_.mx1, input_.vx1, input_.my1, input_.vy1]
data_para_2 = [input_.mx2, input_.vx2, input_.my2, input_.vy2]

## hyperparameter
LearningRate= 0.01


sample_point, A, y = generage_point(data_para_1 = data_para_1, data_para_2 = data_para_2, points_num = input_.N)

### Gradient
w = np.zeros(3)
print("Gradient descent:\n\nw:\n")
Gradient_result_w = Gradient_desent(w = w, A = A, y = y, lr = LearningRate)
gradient_c0, gradient_c1 = predict(w = Gradient_result_w, A = A, sample_point = sample_point, points_num = input_.N)

print("-" * 73)

## Newton's
w = np.zeros(3)
print("Newton's method:\n\nw:\n")
Newton_result_w = Newton(w = w, A = A, y = y, lr = LearningRate)
Newton_c0, Newton_c1 = predict(w = Newton_result_w, A = A, sample_point = sample_point, points_num = input_.N)

ploting(sample_point = sample_point, gradient_c0 = gradient_c0, gradient_c1 = gradient_c1, \
		Newton_c0 = Newton_c0, Newton_c1 = Newton_c1, points_num = input_.N)