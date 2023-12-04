import numpy as np

class Environment():

    def __init__(self):
        self.nS = 5
        self.realNS = self.nS**2
        self.nA = 2
        self.mu = [0,0,1,0,0]
        self.gamma = 0.9
        self.currentStates = [2,2]
        self.start = [2,2]

    # method to set the environment random seed
    def _seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.currentStates = self.start
        s = self.tuple_to_state(self.currentStates)
        return s

    def tuple_to_state(self,tuple):
        s = tuple[0]*self.nS + tuple[1]
        return s

    def transition_model(self,a):
        cs = self.currentStates
        if a[0] == 0 and a[1] == 0:
            self.currentStates = [min(cs[0]+1,4),min(cs[1]+1,4)]
            s_prime = self.currentStates
            reward = -2
        elif a[0] == 1 and a[1] == 1:
            self.currentStates = [max(cs[0]-1,0),max(cs[1]-1,0)]
            s_prime = self.currentStates
            reward = -2
        elif a[0] == 1 and a[1] == 0:
            self.currentStates = [max(cs[0]-1,0),min(cs[1]+1,4)]
            s_prime = self.currentStates
            if (s_prime[0] == 0 and s_prime[1] == 4) or (s_prime[0] == 4 and s_prime[1] == 0):
                reward = 1
            elif s_prime[0]==cs[0] or s_prime[1]==cs[1]:
                reward = -1
            else:
                reward = -0.1
        elif a[0] == 0 and a[1] == 1:
            self.currentStates = [min(cs[0]+1,4),max(cs[1]-1,0)]
            s_prime = self.currentStates
            if (s_prime[0] == 0 and s_prime[1] == 4) or (s_prime[0] == 4 and s_prime[1] == 0):
                reward = 1
            elif s_prime[0]==cs[0] or s_prime[1]==cs[1]:
                reward = -1
            else:
                reward = -0.1
        s_prime = self.tuple_to_state(s_prime)
        return s_prime, reward

def action_to_pair(a):
    if a == 0:
        pair = [0,0]
    elif a == 1:
        pair = [0,1]
    elif a == 2:
        pair = [1,0]
    elif a == 3:
        pair = [1,1]
    return pair

def eps_greedy(s, Q, eps):
    actions = [0,1,2,3]
    if np.random.random() < eps:
        a = np.random.choice(actions)
    else:
        Q_s = Q[s, :].copy()
        a = np.argmax(Q_s)
    return a

def qLearning():
    env = Environment()
    env._seed(15)
    # learning parameters
    M = 60
    m = 1
    k = 3 # length of the episode
    # initial Q function
    Q = np.zeros((env.nS**2,env.nA**2))
    print('Starting...')
    while m<M:
        # initial state and action
        s = env.reset()
        alpha = (1 - m/M)
        eps = (1 - m/M) ** 3
        a = eps_greedy(s, Q, eps)
        # execute an entire episode of three actions
        print('Action execution:')
        for i in range(0,k):
            actions = action_to_pair(a)
            print('s:',env.currentStates)
            s_prime, reward = env.transition_model(actions)
            print('a:',actions)
            print('s_prime:',env.currentStates)
            print('rew:',reward)
            print('*************************')
            # policy improvement step
            a_prime = eps_greedy(s_prime,Q,eps)
            # Q-learning update
            Q[s, a] = Q[s, a] + alpha * (reward + env.gamma * np.max(Q[s_prime, :]) - Q[s, a])
            s = s_prime
            a = a_prime
            if s == 4 or s == 20:
                break
        # next iteration
        print('Iteretion n.',m)
        m = m + 1
        print('Q matrix updated:')
        print(Q)
        print('----------------------------------------------')
    return Q


Q = qLearning()