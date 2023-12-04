import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

class Grid_Environment():

    def __init__(self):
        self.nRows = 5
        self.nCols = 5
        self.nS = self.nRows*self.nCols
        self.nA = 4
        self.nO = 4 # number of ostacle on the grid
        self.start = [0,0]
        self.goal = [4,4]
        self.position = self.start
        # allowed actions is not necessary, the robot can take every action in every state
        self.actual_states = self.nS - self.nO
        mu = []
        for i in range(0,self.actual_states):
            if i == 0:
                mu.append(1.0)
            else:
                mu.append(0.0)
        self.mu = np.array(mu)
        self.gamma = 0.9
        self.grid = np.array([[2,0,0,0,0],[0,1,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[0,1,0,0,3]])


    def set_start(self,start):
        # with this method you can set the starting point of the grid
        # as a tuple of row and column
        self.start = start
        self.grid[start[0],start[1]] = 2

    def set_goal(self,goal):
        # with this method you can set the starting point of the grid
        # as a tuple of row and column
        self.goal = goal
        self.grid[goal[0],goal[1]] = 3

    def set_position(self,pos):
        self.position = pos
    #def create_grid(self):
    def get_position(self):
        return self.position

    def transition_model(self,a):
        # action encoding:
        # 0: up
        # 1: down
        # 2: left
        # 3: right
        eps = 0.1
        s = self.get_position()
        if a == 0:
            if s[0] - 1 >= 0 and self.grid[s[0]-1,s[1]] != 1:
                s_prime = [s[0] - 1,s[1]]
                if self.grid[s_prime[0],s_prime[1]] == 3:
                    inst_rew = 1
                else:
                    inst_rew = -eps
            else:
                s_prime = s
                inst_rew = -1
        elif a == 1:
            if s[0] + 1 < self.nRows and self.grid[s[0]+1,s[1]] != 1:
                s_prime = [s[0] + 1,s[1]]
                if self.grid[s_prime[0],s_prime[1]] == 3:
                    inst_rew = 1
                else:
                    inst_rew = -eps
            else:
                s_prime = s
                inst_rew = -1
        elif a == 2:
            if s[1] - 1 >= 0 and self.grid[s[0],s[1] - 1] != 1:
                s_prime = [s[0],s[1]-1]
                if self.grid[s_prime[0],s_prime[1]] == 3:
                    inst_rew = 1
                else:
                    inst_rew = -eps
            else:
                s_prime = s
                inst_rew = -1
        elif a == 3:
            if s[1] + 1 < self.nCols and self.grid[s[0],s[1]+1] != 1:
                s_prime = [s[0],s[1]+1]
                if self.grid[s_prime[0],s_prime[1]] == 3:
                    inst_rew = 1
                else:
                    inst_rew = -eps
            else:
                s_prime = s
                inst_rew = -1
        return s_prime,inst_rew

    def tuple_to_state(self,t):
        s = t[0]*self.nCols + t[1]
        return s

    def comeback_function(self):
        self.position = self.start

    # method to set the environment random seed
    def _seed(self, seed):
        np.random.seed(seed)

# definition of the greedy policy for our model
def eps_greedy(s, Q, eps):
    if np.random.rand() <= eps:
        actions = np.array([0,1,2,3])
        a = np.random.choice(actions, p=(np.ones(len(actions)) / len(actions)))
    else:
        Q_s = Q[s, :].copy()
        #Q_s[allowed_actions == 0] = - np.inf
        a = np.argmax(Q_s)
    return a

def qLearning():
    env = Grid_Environment()
    env._seed(10)
    # learning parameters
    M = 150
    m = 1
    k = 8 # length of the episode
    # initial Q function
    Q = np.zeros((env.nS, env.nA))
    #first action

    while m <= M:
        env.set_position(env.start)
        s = env.tuple_to_state(env.start)
        a = eps_greedy(s, Q, 1.)
        alpha = (1 - m/M)
        eps = (1 - m/M) ** 2
        for i in range (0,k):
            if m%10 == 0:
                print(m)
                print('step:',i)
                print('s:',s)
                print('a:',a)
            s_prime_tuple, reward = env.transition_model(a)
            s_prime = env.tuple_to_state(s_prime_tuple)
            # policy improvement step
            a_prime = eps_greedy(s_prime,Q,eps)
            # Q-learning update
            Q[s, a] = Q[s, a] + alpha * (reward + env.gamma * np.max(Q[s_prime, :]) - Q[s, a])
            # update the environment position
            env.set_position(s_prime_tuple)
            s = s_prime
            a = a_prime
            #if s == 24:
                #break
        # next iteration
        m = m + 1
        print('Q matrix updated:')
        print(Q)
    print('Final Q function:\n',Q)
    return Q




Q = qLearning()



