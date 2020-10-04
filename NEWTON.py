import numpy as np

class NEWTON(object):
    def __init__(self, A, b):
        self.A = A
        self.b = b
	
    def nEwToN(self, input_x):
        trash, size_A = self.A.shape

        x0 = np.random.random_sample((size_A,))

        #print("x0: ", x0)
        #print("b: ", self.b.shape)
        error = 99.9

        while error > 0.001:
        	ATA = (self.A.T @ self.A)
        	Inv_Hession = np.linalg.inv(2 * ATA)
        	#print("Inv_Hession", Inv_Hession.shape)
        	#print("x0", x0.shape)
        	ATA_x0 = (ATA @ x0)
        	#print("ATA_x0", ATA_x0.shape)
        	ATA_x0_2 = 2 * ATA_x0
        	#print("ATA_x0_2", ATA_x0_2.shape)
        	ATb = (self.A.T @ self.b)
        	#print("ATb", ATb.shape)
        	ATb_2 = 2 * ATb
        	#print("ATb_2", ATb_2.shape)
        	ATA_x0_2_ATb_2 = ATA_x0_2 - ATb_2
        	#print("two:", ATA_x0_2, ATb_2)
        	#print("ATA_x0_2_ATb_2", ATA_x0_2_ATb_2.shape)
        	Inv_Hession_ATA_x0_2_ATb_2 = Inv_Hession @ ATA_x0_2_ATb_2
        	#print("Inv_Hession_ATA_x0_2_ATb_2", Inv_Hession_ATA_x0_2_ATb_2.shape)

        	x1 = x0 - Inv_Hession_ATA_x0_2_ATb_2

        	error = np.sum(np.square(x1 - x0))
        	x0 = x1
        #print(x0.shape)
        loss = self.nEwToN_loss(ans = x0)
        return x0, loss

    def nEwToN_loss(self, ans):
        ## x * ans
        ans_vector = self.A @ ans
        return np.sum(np.square(ans_vector - self.b))