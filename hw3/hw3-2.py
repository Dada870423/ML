import sys
from gaussian import *



mean = float(input("mean: "))
varinance = float(input("varinance: "))

print("Data point source function: N(", mean, ", ", varinance, ")")

sourse = list()

for i in range(10000):
    SSSS = Univariate_generator(mean = mean, variance = varinance)
    print("Add data point: ", SSSS)
    print("mean = ", np.mean(sourse), "   Variance = ", np.var(sourse))
    sourse.append(SSSS)
