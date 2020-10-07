import numpy as np
from MATRIX_OP import MATRIX_OP

class LSE(object):
    def __init__(self, A, b, LambDa):
        self.A = A
        self.b = b
        self.LambDa = LambDa

    def LsE(self, input_x):
        matop = MATRIX_OP()

        trash, size_A = self.A.shape

        ## ATA+lambda I
        ATA = self.A.T @ self.A
        I = np.identity(size_A)
        ATA_Lamda_I = ATA + self.LambDa * I

        Inve_ATA_Lamda_I = matop.iNvErSe(ATA_Lamda_I)


        Inve_ATA_Lamda_I_AT = (Inve_ATA_Lamda_I @ self.A.T)
        Inve_ATA_Lamda_I_ATb = Inve_ATA_Lamda_I_AT @ self.b
        loss = self.Lse_loss(ans = Inve_ATA_Lamda_I_ATb)
        return Inve_ATA_Lamda_I_ATb, loss

    def Lse_loss(self, ans):
        ## x * ans
        ans_vector = self.A @ ans
        return np.sum(np.square(ans_vector - self.b))






