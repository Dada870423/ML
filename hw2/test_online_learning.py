from online_learning import online_learning
#from decimal import *
#print(getcontext())
#Context(prec=28, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999, capitals=1, clamp=0, flags=[], traps=[InvalidOperation, DivisionByZero, Overflow])
import os


ON_learning = online_learning()

ON_learning.Train("test_file.txt")