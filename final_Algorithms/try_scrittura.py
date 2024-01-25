import numpy as np
import sys


Q = np.random.randint(0,10,size = (625,16))
np.set_printoptions(threshold=sys.maxsize)
np.save('/Users/alessandrodonofrio/Desktop/Spike Prime Python/final_Algorithms/qMatrix.npy', Q)

T = np.load('/Users/alessandrodonofrio/Desktop/Spike Prime Python/final_Algorithms/qMatrix.npy')
print(T)