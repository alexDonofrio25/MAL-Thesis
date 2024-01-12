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
            inst_rew = 1
            inst_rew_greedy = inst_rew
            return s_prime,inst_rew, inst_rew_greedy
        else:
            if a == 0:
                s_prime = s - self.nRows
                if s_prime < 0 or inst_rew == -1:
                    s_prime = s
                agent.set_position(s_prime)
            elif a == 1:
                s_prime = s + self.nRows
                if s_prime > 24 or inst_rew == -1:
                    s_prime = s
                agent.set_position(s_prime)
            elif a == 2:
                s_prime = s - 1
                if s_prime < 0 or inst_rew == -1:
                    s_prime = s
                agent.set_position(s_prime)
            elif a == 3:
                s_prime = s + 1
                if s_prime > 24 or inst_rew == -1:
                    s_prime = s
                agent.set_position(s_prime)
            if grid_array[s_prime] == 1:
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
        y = -1
        z = 1
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
def eps_greedy(s1, s2, Q, eps, allowed_actions):
    actions1 = np.where(allowed_actions[s1])
    actions2 = np.where(allowed_actions[s2])
    actions1 = actions1[0] # just keep the indices of the allowed actions
    actions2 = actions2[0]
    s = tuple_to_state([s1,s2])
    Q_s = Q[s, :].copy()
    i = 0
    for qs in Q_s:
        s_prime1 = int(i/4)
        s_prime2 = int(i%4)
        if (s_prime1 not in actions1) or (s_prime2 not in actions2):
            Q_s[i] = -np.inf
        i+=1
    a_greedy = np.argmax(Q_s)
    if np.random.rand() <= eps:
        a1 = np.random.choice(actions1, p=(np.ones(len(actions1)) / len(actions1)))
        a2 = np.random.choice(actions2, p=(np.ones(len(actions2)) / len(actions2)))
        a = tuple_to_action([a1,a2])
    else:
        a = a_greedy
    return a, a_greedy

def tuple_to_state(tuple):
        s = tuple[0]*25 + tuple[1]
        return s

def coupled_action(a):
    action_pair = [int(a/4),int(a%4)]
    return action_pair

def tuple_to_action(tuple):
    action = tuple[0]*4 + tuple[1]
    return action

def update_function(Q, alpha, gamma, s, s_prime, a, r, allowed_actions):
    Q_s = Q[s,:].copy()
    Q_sp = Q[s_prime,:].copy()
    Q_sa = Q[s,a].copy()
    s1_p = int(s_prime/25)
    s2_p = int(s_prime%25)
    actions1 = np.where(allowed_actions[s1_p])
    actions2 = np.where(allowed_actions[s2_p])
    actions1 = actions1[0] # just keep the indices of the allowed actions
    actions2 = actions2[0]
    i = 0
    for qs in Q_sp:
        a1 = int(i/4)
        a2 = int(i%4)
        if (a1 not in actions1) or (a2 not in actions2):
            Q_sp[i] = -np.inf
        i+=1
    Q_max = np.max(Q_sp)
    Q_sa1= Q_sa + alpha * (r + (gamma * Q_max) - Q_sa)
    Q[s,a] = Q_sa1
    return Q


def centralized_qlearning(epochs, ep_length, gamma, seed, eps_mode):
    agent1 = Agent('Spiky',0)
    agent2 = Agent('Roby',4)
    collisions = []

    env = Environment()

    env._seed(seed)
    env.setup()

    # learning parameters
    M = epochs
    m = 0
    k = ep_length # length of the episode
    # initial Q function
    Q = np.zeros((env.nS**2,env.nA**2))

    # Keeps track of useful statistics
    episodes_reward = np.zeros((epochs))
    episodes_reward_greedy = np.zeros((epochs))
    episodes_reward_agents = np.zeros((2,epochs))
    episodes_reward_agents_greedy = np.zeros((2,epochs))

    while m<M:
        alpha = (1 - (m+1)/M)
        #alpha = 1/(m+1)
        if eps_mode == 'epochs':
            eps = (1 - m/M) ** 2
        elif eps_mode == 'quadratic':
            eps = (1/(m+1))**2
        elif eps_mode == 'cubic':
            eps = (1/(m+1))**3
        elif eps_mode == 'exponential':
            a = 0.01
            eps = np.exp(-a*m)
        elif eps_mode == 'trial':
            eps = 1/(m+1)
        # initial state and action
        action_reward1 = []
        action_reward2 = []
        s1 = agent1.get_position()
        s2 = agent2.get_position()
        s = tuple_to_state([s1,s2])
        a, a_greedy = eps_greedy(s1,s2, Q, eps, env.allowed_actions)
        # execute an entire episode of two actions
        for i in range(0,k):
            actions = coupled_action(a)
            actions_greedy = coupled_action(a_greedy)
            s1 = agent1.get_position()
            s2 = agent2.get_position()
            #environment exploration
            s1_prime, r1, r1_greedy = env.transition_model(agent1,actions[0], actions_greedy[0])
            s2_prime, r2, r2_greedy = env.transition_model(agent2,actions[1], actions_greedy[1])
            # collision control
            if s1_prime == s2_prime or (s1 == s2_prime and s2 == s1_prime):
                r = -1
                # agent 1 is the master, so agent2 gets the bad reward
                grid_flatten = env.grid.flatten()
                if grid_flatten[s2] != 3:
                    if s2 != s2_prime:
                        s2_prime = s2
                        agent2.set_position(s2_prime)
                        r2 = r
                    else:
                        s1_prime = s1
                        agent1.set_position(s1_prime)
                        r1 = r
                else:
                    s1_prime = s1
                    agent1.set_position(s1_prime)
                    r1 = r
                collisions.append(m)
            # state generation
            s_prime = tuple_to_state([s1_prime,s2_prime])
            # compute the cumulated reward
            #reward = r1 + r2
            #reward_greedy = r1_greedy + r2_greedy
            reward = np.min([r1,r2])
            reward_greedy = np.min([r1_greedy,r2_greedy])
            # policy improvement step
            a_prime, a_greedy = eps_greedy(s1_prime,s2_prime,Q,eps, env.allowed_actions)
            # Update stats
            action_reward1.append([s1,actions[0],r1])
            action_reward2.append([s2,actions[1],r2])
            episodes_reward[m] += reward
            episodes_reward_greedy[m] += reward_greedy
            episodes_reward_agents[0,m] += r1
            episodes_reward_agents[1,m] += r2
            episodes_reward_agents_greedy[0,m] += r1_greedy
            episodes_reward_agents_greedy[1,m] += r2_greedy
            # Q-learning update
            Q = update_function(Q,alpha,gamma,s,s_prime,a,reward, env.allowed_actions)
            #Q[s, a] = Q[s, a] + alpha * (reward + gamma * np.max(Q[s_prime, :]) - Q[s, a])
            s = s_prime
            a = a_prime
        # next iteration
        m = m + 1
        env.comeback(agent1,0)
        env.comeback(agent2,4)
        print('epochs ', m-1)
        print('Actions/Reward Agent 1: ', action_reward1)
        print('Actions/Reward Agent 2: ', action_reward2)
        #print(collisions)
    #print('Q matrix:')
    #print(Q)
    #print('----------------------------------------------')
    #print(collisions)
    return Q,episodes_reward, episodes_reward_greedy, episodes_reward_agents, episodes_reward_agents_greedy

def confidency_gaps(n,epochs):
    rews = np.zeros((n,epochs))
    for i in range(0,n):
        # ogni esperimento è eseguito con seed diversi
        Q, ep_reward, ep_rew_g, ep_rew_a, ep_rew_ag = centralized_qlearning(epochs, ep_length=6, gamma=0.9, seed=i, eps_mode='exponential')
        # ep_reward matrice 2xM dove M sono le epoche, contiene il reward totale per ogni episodio
        rews[i] = ep_reward

    mean = np.mean(rews, axis=0)
    std = np.std(rews, axis=0)/np.sqrt(n)

    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(10, 5))
    ax.set_title('Agents')
    ax.plot(mean, color='blue')
    ax.fill_between(range(0,len(mean)), (mean - std), (mean + std), alpha = .3)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Rewards')
    ax.set_xticks(np.arange(0,epochs + 1,50))
    plt.show()

def sensitivity_analysis(n, epochs, ep_range, eps_range, greedy):
    fig = plt.figure(figsize=(15,20))
    ax_array = fig.subplots(len(ep_range),len(eps_range), squeeze=False)
    j = 0
    for k in ep_range:
        w = 0
        for eps in eps_range:
            print(k,'/',eps)
            rews = np.zeros((n,epochs))
            g_rews = np.zeros((n,epochs))
            for i in range(0,n):
                Q, ep_rew, ep_g_rew, ep_rew_a, ep_rew_ag = centralized_qlearning(epochs, k,  gamma=0.9, seed= i, eps_mode= eps)
                # ep_reward matrice 2xM dove M sono le epoche, contiene il reward totale per ogni episodio
                rews[i] = ep_rew
                g_rews[i] = ep_g_rew
            if greedy == True:
                mean = np.mean(g_rews, axis = 0)
                std = np.std(g_rews, axis=0)/np.sqrt(n)
                title = str(k)+'/'+eps
                ax_array[j,w].set_title(title)
                ax_array[j,w].set_xlabel('Epochs')
                ax_array[j,w].set_ylabel('Reward')
                ax_array[j,w].plot(mean,color='blue')
                ax_array[j,w].fill_between(range(0,len(mean)), (mean - std), (mean + std), alpha = .3)
                w += 1
                print('----------------------------------------------')
            else:
                mean = np.mean(rews, axis = 0)
                std = np.std(rews, axis=0)/np.sqrt(n)
                title = str(k)+'/'+eps
                ax_array[j,w].set_title(title)
                ax_array[j,w].set_xlabel('Epochs')
                ax_array[j,w].set_ylabel('Reward')
                ax_array[j,w].plot(mean,color='blue')
                ax_array[j,w].fill_between(range(0,len(mean)), (mean - std), (mean + std), alpha = .3)
                w += 1
                print('----------------------------------------------')
        j += 1
    plt.show()

#confidency_gaps(100,1000)

ep_range = [7,8,9]
#eps_range = ['epochs','quadratic','cubic','exponential']
eps_range = ['epochs','trial','quadratic']
sensitivity_analysis(30,500, ep_range, eps_range, True)
Q, ep_rew, ep_rew_g, ep_rew_a, ep_rew_ag = centralized_qlearning(900, 7,0.9, 50, 'trial')
fig = plt.figure()
ax = fig.subplots(1,1)
ax.plot(range(0, len(ep_rew)),ep_rew, color = 'red')
ax.plot(range(0, len(ep_rew_g)),ep_rew_g, color = 'blue')
plt.show