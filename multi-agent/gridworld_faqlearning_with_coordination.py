import numpy as np
import time

class Agent():

    def __init__(self, name, pos):
        self.name = name
        self.position = pos

    def set_position(self,pos):
        self.position = pos

    def get_position(self):
        return self.position

class Environment():
    def __init__(self):
            self.nRows = 5
            self.nCols = 5
            self.nS = self.nRows*self.nCols
            self.nA = 5 # adding the action NOOP
            self.nO = 4 # number of ostacle on the grid
            self.allowed_actions = np.array([[0,1,0,1,1],[0,1,1,1,1],[0,1,1,1,1],[0,1,1,1,1],[0,1,1,0,1],
                                             [1,1,0,1,1],[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1],[1,1,1,0,1],
                                             [1,1,0,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,0,1],
                                             [1,1,0,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,0,0,0,0],[1,1,1,0,1],
                                             [1,0,0,1,1],[0,0,0,0,0],[1,0,1,1,1],[1,0,1,1,1],[1,0,1,0,1]])
            # here the reward given for reaching the target is not maximum, because we try to give the best one when the arget is reached together
            self.R = np.array([[0,-0.1,0,-0.1,-0.5],[0,-1,-0.1,-0.1,-0.5],[0,-1,-0.1,-0.1,-0.5],[0,-0.1,-0.1,-0.1,-0.5],[0,-0.1,-0.1,0,-0.5],
                                [-0.1,-0.1,0,-1,-0.5],[0,0,0,0,0],[0,0,0,0,0],[-0.1,-0.1,-1,-0.1,-0.5],[-0.1,-0.1,-0.1,0,-0.5],
                                [-0.1,-0.1,0,-0.1,-0.5],[-1,-0.1,-0.1,-0.1,-0.5],[-1,0.5,-0.1,-0.1,-0.5],[-0.1,-1,-0.1,-0.1,-0.5],[-0.1,-0.1,-0.1,0,-0.5],
                                [-0.1,-0.1,0,-0.1,-0.5],[-0.1,-1,-0.1,0.5,-0.5],[-1,-1,-1,-1,1],[0,0,0,0,0],[-0.1,-0.1,-1,0,-0.5],
                                [-0.1,0,0,-1,-0.5],[0,0,0,0,0],[-1,-1,-1,-1,1],[-1,0,0.5,-0.1,-0.5],[-0.1,0,-0.1,0,-0.5]])
            self.actual_states = self.nS - self.nO
            mu = []
            for i in range(0,self.actual_states):
                if i == 0:
                    mu.append(1.0)
                else:
                    mu.append(0.0)
            self.mu = mu
            self.gamma = 0.9
            self.grid = np.array([[2,0,0,0,2],
                         [0,1,1,0,0],
                         [0,0,0,0,0],
                         [0,0,3,1,0],
                         [0,1,3,0,0]])

    def transition_model(self,agent,a):

        # action encoding:
        # 0: up
        # 1: down
        # 2: left
        # 3: right
        s = agent.get_position()
        inst_rew = self.R[s,a]
        if a == 0:
            s_prime = s - self.nRows
            agent.set_position(s_prime)
        elif a == 1:
            s_prime = s + self.nRows
            agent.set_position(s_prime)
        elif a == 2:
            s_prime = s - 1
            agent.set_position(s_prime)
        elif a == 3:
            s_prime = s + 1
            agent.set_position(s_prime)
        elif a == 4:
            s_prime = s
            agent.set_position(s_prime)
        grid_array = self.grid.flatten()
        if grid_array[s_prime] == 1:
            s_prime = s
            agent.set_position(s_prime)
        return s_prime,inst_rew

    def tuple_to_state(self,t):
        s = t[0]*self.nCols + t[1]
        return s

    def comeback(self, agent, pos):
        agent.set_position(pos)

    # method to set the environment random seed
    def _seed(self, seed):
        np.random.seed(seed)

# definition of the greedy policy for our model
def eps_greedy(s, Q, eps, allowed_actions):
  if np.random.rand() <= eps:
    actions = np.where(allowed_actions[s])
    actions = actions[0] # just keep the indices of the allowed actions
    a = np.random.choice(actions, p=(np.ones(len(actions)) / len(actions)))
    xi = eps
  else:
    mult = len(np.where(allowed_actions[s][0]))
    Q_s = Q[s, :].copy()
    Q_s[allowed_actions[s] == 0] = - np.inf
    a = np.argmax(Q_s)
    xi = 1 - (mult-1)*eps
  return a, xi

def faq_qlearning():
    spiky = Agent('Spiky',0)
    roby = Agent('Roby',4)
    collisions = []
    env1 = Environment()
    env2 = Environment()

    env1._seed(10)
    env2._seed(13)
    # learning parameters
    M = 250
    m = 1
    k = 7 # length of the episode
    beta = 0.8
    # initial Q function
    Q1 = np.zeros((env1.nS,env1.nA))
    Q2 = np.zeros((env2.nS,env2.nA))

    while m<M:
        print('iteretion n.',m)
        eps = (1 - m/M) ** 2
        alpha = (1 - m/M)
        # initial state and action
        s1 = spiky.get_position()
        s2 = roby.get_position()
        a1,xi1 = eps_greedy(s1, Q1, eps, env1.allowed_actions)
        a2,xi2 = eps_greedy(s2, Q2, eps, env2.allowed_actions)
        # execute an entire episode of two actions
        for i in range(0,k):
            s_prime1, reward1 = env1.transition_model(spiky,a1)
            s_prime2, reward2 = env2.transition_model(roby,a2)
            # check if the robots collide
            if s_prime1 == s_prime2:
                print('Collision!')
                collisions.append(m)
                if s_prime1 == 17 or s_prime1 == 22 or s_prime2 == 17 or s_prime2 == 22:
                    if s_prime1 == 17 or s_prime1 == 22:
                        flag = 1
                    else:
                        flag = 2
                else:
                    flag = np.random.choice([1,2])

                if flag == 1:
                    s_prime2 = s2
                    reward2 = -1
                else:
                    s_prime1 = s1
                    reward1 = -1
            # check if both agents reach a target in the next iteration:
            if (s_prime1 != s1) and (s_prime2 != s2):
                if (s_prime1 == 17 or s_prime1 == 22) and (s_prime2 == 17 or s_prime2== 22):
                    print('Target reached together!')
                    reward1 = 1.0
                    reward2 = 1.0
            # Q-learning update
            Q1[s1, a1] = Q1[s1, a1] + np.min([beta/xi1,1]) * alpha*(reward1 + env1.gamma * np.max(Q1[s_prime1, :]) - Q1[s1, a1])
            Q2[s2, a2] = Q2[s2, a2] + np.min([beta/xi2,1]) * alpha*(reward2 + env2.gamma * np.max(Q2[s_prime2, :]) - Q2[s2, a2])
            # policy improvement step
            a_prime1,xi1 = eps_greedy(s_prime1,Q1,eps, env1.allowed_actions)
            a_prime2,xi2 = eps_greedy(s_prime2,Q2,eps, env2.allowed_actions)
            s1 = s_prime1
            a1 = a_prime1
            s2 = s_prime2
            a2 = a_prime2
            if (s1 == 17 or s1 == 22) and (s2 == 17 or s2 == 22):
                break
        # next iteration

        m = m + 1
        env1.comeback(spiky,0)
        env2.comeback(roby,4)
        print('Q1 matrix updated:')
        print(Q1)
        print('----------------------------------------------')
        print('Q2 matrix updated:')
        print(Q2)
        print('----------------------------------------------')
        print(collisions)
    return Q1,Q2

q1,q2 = faq_qlearning()