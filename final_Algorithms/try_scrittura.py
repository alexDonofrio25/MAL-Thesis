import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
Q = np.load('/Users/alessandrodonofrio/Desktop/Spike Prime Python/final_Algorithms/qMatrix_340.npy')
for i in range(0,21):
    if i%10 == 0:
        path = '/Users/alessandrodonofrio/Desktop/Spike Prime Python/final_Algorithms/'
        file = 'qMatrix_' + str(i) + '.npy'
        complete_path = path+file
        f = open(complete_path, 'a')
        np.save(complete_path, Q)

