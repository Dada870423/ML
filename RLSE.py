import numpy as np


class RLSE(object):
    def __init__(self, A, b, LambDa):
        self.A = A
        self.b = b
        self.LambDa = LambDa

    def rLsE(self, input_x):
        trash, size_A = self.A.shape

        ## ATA+lambda I
        ATA = self.A.T @ self.A
        I = np.identity(size_A)
        ATA_Lamda_I = ATA + self.LambDa * I
        #print("ATA_Lamda_I", ATA_Lamda_I.shape)

        Inve_ATA_Lamda_I = np.linalg.inv(ATA_Lamda_I)
        #print("Inve_ATA_Lamda_I", Inve_ATA_Lamda_I.shape)
        Inve_ATA_Lamda_I_AT = (Inve_ATA_Lamda_I @ self.A.T)
        #print("Inve_ATA_Lamda_I_AT", Inve_ATA_Lamda_I_AT.shape)
        Inve_ATA_Lamda_I_ATb = Inve_ATA_Lamda_I_AT @ self.b
        #print("Inve_ATA_Lamda_I_ATb", Inve_ATA_Lamda_I_ATb.shape)
        loss = self.rLse_loss(ans = Inve_ATA_Lamda_I_ATb)
        #print("Inve_ATA_Lamda_I_ATb", Inve_ATA_Lamda_I_ATb)
        return Inve_ATA_Lamda_I_ATb, loss

    def rLse_loss(self, ans):
        ## x * ans
        ans_vector = self.A @ ans
        return np.sum(np.square(ans_vector - self.b))






