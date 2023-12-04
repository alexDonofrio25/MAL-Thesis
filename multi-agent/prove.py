import numpy as np

grid = np.array([[2,0,0,0,2],
                         [0,1,1,0,0],
                         [0,0,0,0,0],
                         [0,0,0,1,0],
                         [3,1,0,0,3]])

print(grid.flatten())
allowed_actions = np.array([[0,1,0,1],[0,1,1,1],[0,1,1,1],[0,1,1,1],[0,1,1,0],
                                             [1,1,0,1],[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,0],
                                             [1,1,0,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,0],
                                             [1,1,0,1],[1,1,1,1],[1,1,1,1],[0,0,0,0],[1,1,1,0],
                                             [1,0,0,1],[0,0,0,0],[1,0,1,1],[1,0,1,1],[1,0,1,0]])
print(allowed_actions[11])


def coupled_action(a):
        action_pair = [int(a/4),int(a%4)]
        return action_pair

print(coupled_action(9))