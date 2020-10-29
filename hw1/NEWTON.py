import numpy as np
from MATRIX_OP import MATRIX_OP
class NEWTON(object):
    def __init__(self, A, b):
        self.A = A
        self.b = b
	
    def nEwToN(self, input_x):
        matop = MATRIX_OP()
        trash, size_A = self.A.shape

        x0 = np.random.random_sample((size_A,))

        error = 99.9

        while error > 0.1:
            ATA = (self.A.T @ self.A)
            Inv_Hession = matop.iNvErSe(2 * ATA)
            ATA_x0 = (ATA @ x0)
            ATA_x0_2 = 2 * ATA_x0
            ATb = (self.A.T @ self.b)
            ATb_2 = 2 * ATb
            ATA_x0_2_ATb_2 = ATA_x0_2 - ATb_2
            Inv_Hession_ATA_x0_2_ATb_2 = Inv_Hession @ ATA_x0_2_ATb_2

            x1 = x0 - Inv_Hession_ATA_x0_2_ATb_2

            error = np.sum(np.square(x1 - x0))
            x0 = x1
        loss = self.nEwToN_loss(ans = x0)
        return x0, loss

    def nEwToN_loss(self, ans):
        ## x * ans
        ans_vector = self.A @ ans
        return np.sum(np.square(ans_vector - self.b))