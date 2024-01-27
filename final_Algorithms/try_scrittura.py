import numpy as np
import sys

Q = np.random.randint((10,5))
for i in range(0,21):
    if i%10 == 0:
        path = '/Users/alessandrodonofrio/Desktop/Spike Prime Python/final_Algorithms/'
        file = 'qMatrix_' + str(i) + '.npy'
        complete_path = path+file
        f = open(complete_path, 'a')
        np.save(complete_path, Q)

