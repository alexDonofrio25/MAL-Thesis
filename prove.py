import numpy as np
import matplotlib.pyplot as plt

R = np.zeros((25,4))
all_actions = np.zeros((25,4))
grid = np.array([[2,0,0,0,2],
                         [0,1,1,0,0],
                         [0,0,0,0,0],
                         [0,0,3,1,0],
                         [0,1,3,0,0]])

def r_generator(R):
    for i in range(0,5):
        for j in range(0,5):
            for a in range(0,4):
                if a == 0:
                    if i-1 >= 0:
                        pos = grid[i-1,j]
                        if pos == 0 or pos == 2:
                            R[i*5+j,a] = -0.1
                        elif pos == 1:
                            R[i*5+j,a] = -1
                        elif pos == 3:
                            R[i*5+j,a] = 1
                    else:
                        R[i*5+j,a] = 0
                elif a == 1:
                    if i+1 <= 4:
                        pos = grid[i+1,j]
                        if pos == 0 or pos == 2:
                            R[i*5+j,a] = -0.1
                        elif pos == 1:
                            R[i*5+j,a] = -1
                        elif pos == 3:
                            R[i*5+j,a] = 1
                    else:
                        R[i,a] = 0
                elif a == 2:
                    if j-1 >= 0:
                        pos = grid[i,j-1]
                        if pos == 0 or pos == 2:
                            R[i*5+j,a] = -0.1
                        elif pos == 1:
                            R[i*5+j,a] = -1
                        elif pos == 3:
                            R[i*5+j,a] = 1
                    else:
                        R[i,a] = 0
                elif a == 3:
                    if j+1<=4:
                        pos = grid[i,j+1]
                        if pos == 0 or pos == 2:
                            R[i*5+j,a] = -0.1
                        elif pos == 1:
                            R[i*5+j,a] = -1
                        elif pos == 3:
                            R[i*5+j,a] = 1
                    else:
                        R[i*5+j,a] = 0
    return R


def all_act_generator(allowed_actions):
    for i in range(0,5):
        for j in range(0,5):
            for a in range(0,4):
                if a == 0:
                    pos = i - 1
                    if pos < 0:
                        allowed_actions[i*5+j,a] = 0
                    else:
                        allowed_actions[i*5+j,a] = 1
                elif a == 1:
                    pos = i + 1
                    if pos > 4:
                        allowed_actions[i*5+j,a] = 0
                    else:
                        allowed_actions[i*5+j,a] = 1
                elif a == 2:
                    pos = j -1
                    if pos < 0:
                        allowed_actions[i*5+j,a] = 0
                    else:
                        allowed_actions[i*5+j,a] = 1
                elif a == 3:
                    pos = j + 1
                    if pos > 4:
                        allowed_actions[i*5+j,a] = 0
                    else:
                        allowed_actions[i*5+j,a] = 1

    return allowed_actions
#al = all_act_generator(all_actions)
#r = r_generator(R)
#print(r)
eps = np.zeros(1000)
for m in range(0,1000):
    val = np.exp(-0.01*m)
    eps[m] = val
plt.plot(eps)
plt.show()
print(0.9*np.exp(-2*100))
print(np.sqrt(100))