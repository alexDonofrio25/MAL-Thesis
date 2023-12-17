import numpy as np
import time
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
            self.allowed_actions = np.array([[0,1,0,1],[0,1,1,1],[0,1,1,1],[0,1,1,1],[0,1,1,0],
                                             [1,1,0,1],[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,0],
                                             [1,1,0,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,0],
                                             [1,1,0,1],[1,1,1,1],[1,1,1,1],[0,0,0,0],[1,1,1,0],
                                             [1,0,0,1],[0,0,0,0],[1,0,1,1],[1,0,1,1],[1,0,1,0]])
            self.R = np.array([[0,-0.1,0,-0.1],[0,-1,-0.1,-0.1],[0,-1,-0.1,-0.1],[0,-0.1,-0.1,-0.1],[0,-0.1,-0.1,0],
                                [-0.1,-0.1,0,-1],[0,0,0,0],[0,0,0,0],[-0.1,-0.1,-1,-0.1],[-0.1,-0.1,-0.1,0],
                                [-0.1,-0.1,0,-0.1],[-1,-0.1,-0.1,-0.1],[-1,1,-0.1,-0.1],[-0.1,-1,-0.1,-0.1],[-0.1,-0.1,-0.1,0],
                                [-0.1,-0.1,0,-0.1],[-0.1,-1,-0.1,1],[0,0,0,0],[0,0,0,0],[-0.1,-0.1,-1,0],
                                [-0.1,0,0,-1],[0,0,0,0],[0,0,0,0],[-1,0,1,-0.1],[-0.1,0,-0.1,0]])
            self.actual_states = self.nS - self.nO
            mu = []
            for i in range(0,self.actual_states):
                if i == 0:
                    mu.append(1.0)
                else:
                    mu.append(0.0)
            self.mu = mu
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
        grid_array = self.grid.flatten()
        if grid_array[s_prime] == 1 or grid_array[s] == 3 :
            s_prime = s
            agent.set_position(s_prime)
        return s_prime,inst_rew, inst_rew_greedy

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
    Q_s = Q[s, :].copy()
    Q_s[allowed_actions[s] == 0] = - np.inf
    a_greedy = np.argmax(Q_s)
    actions = np.where(allowed_actions[s])
    actions = actions[0] # just keep the indices of the allowed actions
    if np.random.rand() <= eps:
        a = np.random.choice(actions, p=(np.ones(len(actions)) / len(actions)))
        xi = eps
    else:
        mult = len(actions)
        a = np.argmax(Q_s)
        xi = 1 - (mult-1)*eps
    return a, xi, a_greedy

def faq_learning(epochs, ep_length, beta, gamma, seed1, seed2, eps_mode):
    spiky = Agent('Spiky',0)
    roby = Agent('Roby',4)
    collisions = []
    env1 = Environment()
    env2 = Environment()
    # they generare the allowed actions for the grid and the reward distribution
    #env1.setup()
    #env2.setup()
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

    while m < M:
        # keeps track of the episode's rewards
        rewards = np.zeros((2,k))
        # keeps track of the greedy episode's reward
        rewards_greedy = np.zeros((2,k))

        if eps_mode == 'epochs':
            eps = (1 - m/M) ** 2
        elif eps_mode == 'quadratic':
            eps = (1/(m+1))**2
        elif eps_mode == 'cubic':
            eps = (1/(m+1))**3
        elif eps_mode == 'exponential':
            a = 0.01
            eps = np.exp(-a*m)

        alpha = (1 - m/M)
        #alpha = 0.1
        # initial state and action
        s1 = spiky.get_position()
        s2 = roby.get_position()
        a1,xi1, a1_greedy = eps_greedy(s1, Q1, eps, env1.allowed_actions)
        a2,xi2, a2_greedy = eps_greedy(s2, Q2, eps, env2.allowed_actions)
        # execute an entire episode of two actions
        for i in range(0,k):
            s_prime1, reward1, r1_greedy = env1.transition_model(spiky,a1,a1_greedy)
            s_prime2, reward2, r2_greedy = env2.transition_model(roby,a2,a2_greedy)
            # check if the robots collide
            if s_prime1 == s_prime2:
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
            # Save stats
            rewards[0,i] = reward1
            rewards[1,i] = reward2
            rewards_greedy[0,i] = r1_greedy
            rewards_greedy[1,i] = r2_greedy

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

        # update stats
        episodes_reward[0,m] = np.sum(rewards[0,:])
        episodes_reward[1,m] = np.sum(rewards[1,:])
        ep_greedy_reward[0,m] = np.sum(rewards_greedy[0,:])
        ep_greedy_reward[1,m] = np.sum(rewards_greedy[1,:])

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
    return Q1,Q2, episodes_reward, ep_greedy_reward

def confidency_gaps(n):
    rews = np.zeros((n,2,400))
    for i in range(0,n):
        # ogni esperimento è eseguito con seed diversi
        Q1,Q2, ep_reward, ep_g_reward= faq_learning(epochs=400, ep_length=7, beta=0.7,gamma=0.9, seed1=i, seed2= i + n,eps_mode='cubic')
        # ep_reward matrice 2xM dove M sono le epoche, contiene il reward totale per ogni episodio
        rews[i] = ep_reward
    mean = np.mean(rews, axis = 0)
    std = np.std(rews, axis=0)/np.sqrt(n)

    fig, axs = plt.subplots(nrows=1,ncols=2, figsize=(15, 5))
    axs[0].set_title('Agent 1')
    axs[1].set_title('Agent 2')
    axs[0].plot(mean[0,:], color='blue')
    axs[0].fill_between(range(0,len(mean[0,:])), (mean[0,:] - std[0,:]), (mean[0,:] + std[0,:]), alpha = .3)
    axs[1].plot(mean[1,:], color='red')
    axs[1].fill_between(range(0,len(mean[0,:])), (mean[1,:] - std[1,:]), (mean[1,:] + std[1,:]), alpha = .3, color='red')
    axs[0].set_xlabel('Epochs')
    axs[1].set_xlabel('Epochs')
    axs[0].set_ylabel('Rewards')
    axs[1].set_ylabel('Rewards')
    axs[0].set_xticks(np.arange(0,401,50))
    axs[1].set_xticks(np.arange(0,401,50))
    plt.show()

def sensitivity_analysis(n,ep_range):
    l = len(ep_range)
    rews = np.zeros((n,l,2))
    i = 0
    for e in ep_range:
        for j in range(0,n):
            Q1,Q2,ep_rew = faq_learning(epochs=e, ep_length=7, beta=0.8, gamma=0.9,seed1=j,seed2=j+1)
            max_rew = ep_rew[:,e-1]
            rews[j,i] = max_rew
        i += 1
    mean = np.mean(rews, axis=0)
    std = np.std(rews, axis=0)/n
    #fig, axs = plt.subplots(1,2, figsize = (15,5))
    fig, ax = plt.subplots(figsize = (10,5))
    #axs[0].set_title('Agent 1')
    #axs[1].set_title('Agent 2')
    #axs[0].plot(mean[:,0], color='blue')
    ax.plot(mean[:,0], color='blue')
    ax.plot(mean[:,1], color='red')
    ax.fill_between(range(0,len(mean[:,0])), (mean[:,0] - std[:,0]), (mean[:,0] + std[:,0]), alpha = .3)
    ax.fill_between(range(0,len(mean[:,0:])), (mean[:,1] - std[:,1]), (mean[:,1] + std[:,1]), alpha = .3, color='red')
    #axs[0].fill_between(range(0,len(mean[:,0])), (mean[:,0] - std[:,0]), (mean[:,0] + std[:,0]), alpha = .3)
    #axs[1].plot(mean[:,1], color='red')
    #axs[1].fill_between(range(0,len(mean[:,0:])), (mean[:,1] - std[:,1]), (mean[:,1] + std[:,1]), alpha = .3, color='red')
    #axs[0].set_xlabel('Epochs')
    #axs[1].set_xlabel('Epochs')
    #axs[0].set_ylabel('Rewards')
    #axs[1].set_ylabel('Rewards')
    plt.show()



#q1,q2,ep_rewards,ep_means,ep_std = multi_agent_qlearning(epochs=200,ep_length=7,beta=0.8,gamma=0.9, seed1=12, seed2=15)
#fig, (ax1,ax2)= plt.subplots(ncols=2,figsize=(15, 5))

q1,q2, ep_rewards, ep_g_rewards = faq_learning(400,7,0.7,0.9,1,100,'cubic')
print('Ep_rewards:', ep_rewards)
print('----------------------------------------------')
print('Greedy rewards:', ep_g_rewards)
print(q1)
print(q2)

#confidency_gaps(100)
#e_range = range(100, 450, 50)
#sensitivity_analysis(10,e_range)

