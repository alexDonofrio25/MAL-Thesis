import numpy as np

act1 = np.array([1,3])
act2 = np.array([2,3])

for i in range(0,16):
    if (int(i/4) not in act1) or (int(i%4) not in act2):
        print('Ok')