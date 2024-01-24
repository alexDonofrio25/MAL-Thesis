import numpy as np
import matplotlib.pyplot as plt

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
            self.nA = 4
            self.nO = 4 # number of ostacle on the grid
            self.allowed_actions = np.zeros((self.nS,self.nA))
            self.R = np.zeros((self.nS,self.nA))
            self.actual_states = self.nS - self.nO
            self.grid = np.array([[2,0,0,0,2],
                         [0,1,1,0,0],
                         [0,0,0,0,0],
                         [0,0,3,1,0],
                         [0,1,3,0,0]])

    def transition_model(self,agent,a, a_greedy):
        # action encoding:
        # 0: up
        # 1: down
        # 2: left
        # 3: right
        s = agent.get_position()
        inst_rew = self.R[s,a]
        inst_rew_greedy = self.R[s,a_greedy]
        grid_array = self.grid.flatten()
        if grid_array[s] == 3:
            s_prime = s
            inst_rew = 1.0
            inst_rew_greedy = inst_rew
            return s_prime,inst_rew, inst_rew_greedy
        else:
            if a == 0:
                s_prime = s - self.nRows
                if s_prime < 0:
                    s_prime = s
                agent.set_position(s_prime)
            elif a == 1:
                s_prime = s + self.nRows
                if s_prime > 24:
                    s_prime = s
                agent.set_position(s_prime)
            elif a == 2:
                s_prime = s - 1
                if s_prime < 0:
                    s_prime = s
                agent.set_position(s_prime)
            elif a == 3:
                s_prime = s + 1
                if s_prime > 24:
                    s_prime = s
                agent.set_position(s_prime)
            if grid_array[s_prime] == 1 or grid_array[s] == 3 :
                s_prime = s
                agent.set_position(s_prime)
            return s_prime,inst_rew, inst_rew_greedy

    def tuple_to_state(self,t):
        s = t[0]*self.nCols + t[1]
        return s

    def comeback(self, agent, pos):
        agent.set_position(pos)

    # checked-working
    def r_generator(self):
        x = -0.1
        y = -1.0
        z = 1.0
        for i in range(0,self.nRows):
            for j in range(0,self.nCols):
                for a in range(0,self.nA):
                    if a == 0:
                        if i-1 >= 0:
                            pos = self.grid[i-1,j]
                            if pos == 0 or pos == 2:
                                self.R[i*self.nRows+j,a] = x
                            elif pos == 1:
                                self.R[i*self.nRows+j,a] = y
                            elif pos == 3:
                                self.R[i*self.nRows+j,a] = z
                        else:
                            self.R[i*self.nRows+j,a] = 0
                    elif a == 1:
                        if i+1 <= self.nRows - 1:
                            pos = self.grid[i+1,j]
                            if pos == 0 or pos == 2:
                                self.R[i*self.nRows+j,a] = x
                            elif pos == 1:
                                self.R[i*self.nRows+j,a] = y
                            elif pos == 3:
                                self.R[i*self.nRows+j,a] = z
                        else:
                            self.R[i*self.nRows+j,a] = 0
                    elif a == 2:
                        if j-1 >= 0:
                            pos = self.grid[i,j-1]
                            if pos == 0 or pos == 2:
                                self.R[i*self.nRows+j,a] = x
                            elif pos == 1:
                                self.R[i*self.nRows+j,a] = y
                            elif pos == 3:
                                self.R[i*self.nRows+j,a] = z
                        else:
                            self.R[i*self.nRows+j,a] = 0
                    elif a == 3:
                        if j+1<=self.nCols - 1:
                            pos = self.grid[i,j+1]
                            if pos == 0 or pos == 2:
                                self.R[i*self.nRows+j,a] = x
                            elif pos == 1:
                                self.R[i*self.nRows+j,a] = y
                            elif pos == 3:
                                self.R[i*self.nRows+j,a] = z
                        else:
                            self.R[i*self.nRows+j,a] = 0

     # checked-working
    def all_act_generator(self):
        for i in range(0,self.nRows):
            for j in range(0,self.nCols):
                for a in range(0,self.nA):
                    if a == 0:
                        pos = i - 1
                        if pos < 0:
                            self.allowed_actions[i*self.nRows+j,a] = 0
                        else:
                            self.allowed_actions[i*self.nRows+j,a] = 1
                    elif a == 1:
                        pos = i + 1
                        if pos > self.nRows-1:
                            self.allowed_actions[i*self.nRows+j,a] = 0
                        else:
                            self.allowed_actions[i*self.nRows+j,a] = 1
                    elif a == 2:
                        pos = j - 1
                        if pos < 0:
                            self.allowed_actions[i*self.nRows+j,a] = 0
                        else:
                            self.allowed_actions[i*self.nRows+j,a] = 1
                    elif a == 3:
                        pos = j + 1
                        if pos > self.nCols-1:
                            self.allowed_actions[i*self.nRows+j,a] = 0
                        else:
                            self.allowed_actions[i*self.nRows+j,a] = 1

    def setup(self):
        self.r_generator()
        self.all_act_generator()

    # method to set the environment random seed
    def _seed(self, seed):
        np.random.seed(seed)

# definition of the greedy policy for our model
def eps_greedy(s, Q, eps, allowed_actions):
    Q_s = Q[s, :].copy()
    Q_s[allowed_actions[s] == 0] = - np.inf
    a_greedy = np.argmax(Q_s)
    actions = np.where(allowed_actions[s])
    actions = actions[0] # just keep the indices of the allowed actions
    if np.random.rand() <= eps:
        mult = len(actions)
        a = np.random.choice(actions, p=(np.ones(len(actions)) / len(actions)))
        xi = eps/(mult-1)
    else:
        a = np.argmax(Q_s)
        xi = 1 - eps
    return a, xi, a_greedy

def faq_learning(epochs, ep_length, beta, gamma, seed1, seed2, eps_mode):
    spiky = Agent('Spiky',0)
    roby = Agent('Roby',4)
    collisions = []
    env1 = Environment()
    env2 = Environment()
    # they generare the allowed actions for the grid and the reward distribution
    env1.setup()
    env2.setup()
    # randomize the experiment
    env1._seed(seed1)
    env2._seed(seed2)
    # learning parameters
    M = epochs
    m = 0
    k = ep_length # length of the episode
    # initial Q function
    Q1 = np.zeros((env1.nS,env1.nA))
    Q2 = np.zeros((env2.nS,env2.nA))

    # Keeps track of useful statistics
    episodes_reward = np.zeros((2,epochs))
    ep_greedy_reward = np.zeros((2,epochs))
    ep_rew_joint = np.zeros(epochs)
    ep_rewg_joint = np.zeros(epochs)


    while m < M:

        if eps_mode == 'epochs':
            eps = (1 - m/M) ** 2
        elif eps_mode == 'quadratic':
            eps = (1/(m+1))**2
        elif eps_mode == 'cubic':
            eps = (1/(m+1))**3
        elif eps_mode == 'trial':
            eps = (1/(m+1))**(2/3)

        alpha = (1 - (m+1)/M)
        #alpha = 0.1
        # initial state and action
        action_reward1 = []
        action_reward2 = []
        s1 = spiky.get_position()
        s2 = roby.get_position()
        a1,xi1, a1_greedy = eps_greedy(s1, Q1, eps, env1.allowed_actions)
        a2,xi2, a2_greedy = eps_greedy(s2, Q2, eps, env2.allowed_actions)
        # execute an entire episode of two actions
        for i in range(0,k):
            s_prime1, reward1, r1_greedy = env1.transition_model(spiky,a1,a1_greedy)
            if s_prime1 == s2:
                s_prime1 == s1
                reward1 = -1.0
                spiky.set_position(s_prime1)
                collisions.append(m)
            s_prime2, reward2, r2_greedy = env2.transition_model(roby,a2,a2_greedy)
            if s_prime2 == s_prime1:
                s_prime2 = s2
                reward2 = -1.0
                roby.set_position(s_prime2)
                collisions.append(m)
            # Update stats
            action_reward1.append([s1,a1,reward1])
            action_reward2.append([s2,a2,reward2])
            episodes_reward[0,m] += reward1
            episodes_reward[1,m] += reward2
            ep_greedy_reward[0,m] += r1_greedy
            ep_greedy_reward[1,m] += r2_greedy
            ep_rew_joint[m] += np.min([reward1,reward2])
            ep_rewg_joint[m] += np.min([r1_greedy,r2_greedy])

            # Q-learning update
            Q1[s1, a1] = Q1[s1, a1] + np.min([beta/xi1,1]) * alpha * (reward1 + gamma * np.max(Q1[s_prime1, :]) - Q1[s1, a1])
            Q2[s2, a2] = Q2[s2, a2] + np.min([beta/xi2,1]) * alpha * (reward2 + gamma * np.max(Q2[s_prime2, :]) - Q2[s2, a2])
            # policy improvement step
            a_prime1,xi1, a1_greedy = eps_greedy(s_prime1,Q1,eps, env1.allowed_actions)
            a_prime2,xi2, a2_greedy = eps_greedy(s_prime2,Q2,eps, env2.allowed_actions)
            s1 = s_prime1
            a1 = a_prime1
            s2 = s_prime2
            a2 = a_prime2
            if (s1 == 17 or s1 == 22) and (s2 == 17 or s2 == 22):
                break
        print('epochs ', m)
        print('Actions/Reward Agent 1: ', action_reward1)
        print('Actions/Reward Agent 2: ', action_reward2)
        # next iteration
        m = m + 1
        env1.comeback(spiky,0)
        env2.comeback(roby,4)
    #print('Q1 matrix:')
    #print(Q1)
    #print('----------------------------------------------')
    #print('Q2 matrix:')
    #print(Q2)
    #print('----------------------------------------------')
    #print('Collisions: ',collisions)
    return Q1,Q2, episodes_reward, ep_greedy_reward, ep_rew_joint, ep_rewg_joint



faq_learning(80,7,0.6,0.9,1,101,'quadratic')