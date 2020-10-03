import numpy as np
from scipy.linalg import lu

class RLSE(object):
    def __init__(self, A, b, LambDa):
        self.A = A
        self.b = b
        self.LambDa = LambDa

    def rLsE(self):
    	trash, size_A = self.A.shape
    	#print(self.A)
    	print("size_A", self.A.shape)
    	#print("size_A_T", self.A.T.shape)
    	#print("b_shape", self.b.shape)
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
    	P, L, Upper = lu(ATA_Lamda_I)
    	Lowwer = P @ L
    	A_inverse = np.linalg.inv(Upper) @ np.linalg.inv(Lowwer)
    	loss = Inve_ATA_Lamda_I_ATb * Inve_ATA_Lamda_I_ATb
    	print(loss.sum())
    	print("Inve_ATA_Lamda_I_ATb", Inve_ATA_Lamda_I_ATb)
    	return Inve_ATA_Lamda_I_ATb, Inve_ATA_Lamda_I_ATb
    	#return Inve_ATA_Lamda_I_ATb, np.sum(np.square((self.b - Inve_ATA_Lamda_I_ATb))), Inve_ATA_Lamda_I_ATb
    	#return Inve_ATA_Lamda_I_ATb, ((Inve_ATA_Lamda_I_ATb - self.b)*(Inve_ATA_Lamda_I_ATb - self.b)).sum(), Inve_ATA_Lamda_I_ATb

