import numpy as np
import sys


Q = np.random.randint(0,10,size = (625,16))
np.set_printoptions(threshold=sys.maxsize)
sQ = np.array2string(Q)
with open('/Users/alessandrodonofrio/Desktop/Spike Prime Python/final_Algorithms/qValues.txt', 'a') as f:
    f.write('Ciao1\2')
    f.write(sQ)
    f.close