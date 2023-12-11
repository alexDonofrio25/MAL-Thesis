import numpy as np
import matplotlib.pyplot as plt

class Agents():

    def __init__(self, n1, n2, pos):
        self.name1 = n1
        self.name2 = n2
        self.position = pos

    def set_position(self,pos):
        self.position = pos

    def get_position(self):
        return self.position

class Environment():
    def __init__(self):
            self.nRows = 5
            self.nCols = 5
            self.nS = (self.nRows*self.nCols)**2
            self.nA = 4**2
            self.nO = 4 # number of ostacle on the grid
            self.gamma = 0.9
            self.allowed_actions = np.array([[0,1,0,1],[0,1,1,1],[0,1,1,1],[0,1,1,1],[0,1,1,0],
                                             [1,1,0,1],[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,0],
                                             [1,1,0,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,0],
                                             [1,1,0,1],[1,1,1,1],[1,1,1,1],[0,0,0,0],[1,1,1,0],
                                             [1,0,0,1],[0,0,0,0],[1,0,1,1],[1,0,1,1],[1,0,1,0]])
            self.R = np.array([[0,-0.1,0,-0.1],[0,-1,-0.1,-0.1],[0,-1,-0.1,-0.1],[0,-0.1,-0.1,-0.1],[0,-0.1,-0.1,0],
                                [-0.1,-0.1,0,-1],[0,0,0,0],[0,0,0,0],[-0.1,-0.1,-1,-0.1],[-0.1,-0.1,-0.1,0],
                                [-0.1,-0.1,0,-0.1],[-1,-0.1,-0.1,-0.1],[-1,1,-0.1,-0.1],[-0.1,-1,-0.1,-0.1],[-0.1,-0.1,-0.1,0],
                                [-0.1,-0.1,0,-0.1],[-0.1,-1,-0.1,1],[1,1,1,1],[0,0,0,0],[-0.1,-0.1,-1,0],
                                [-0.1,0,0,-1],[0,0,0,0],[1,1,1,1],[-1,0,1,-0.1],[-0.1,0,-0.1,0]])
            self.grid = np.array([[2,0,0,0,2],
                         [0,1,1,0,0],
                         [0,0,0,0,0],
                         [0,0,3,1,0],
                         [0,1,3,0,0]])



    def reward_generator(self,s1,s2,a1,a2):
        r1 = self.R[s1,a1]
        r2 = self.R[s2,a2]
        reward = r1+r2
        return reward

    def transition_model(self,agents,a):
        # action encoding:
        # 0: up
        # 1: down
        # 2: left
        # 3: right
        collision_flag = False
        pos = agents.get_position()
        s = tuple_to_state(pos)
        actions = coupled_action(a)
        inst_rew = self.reward_generator(pos[0],pos[1],actions[0],actions[1])
        grid_list = self.grid.flatten()
        pos_prime = [0,0]
        if actions[0] == 0:
            pos_prime[0] = pos[0] - self.nRows
        elif actions[0] == 1:
            pos_prime[0] = pos[0] + self.nRows
        elif actions[0] == 2:
            pos_prime[0] = pos[0] - 1
        elif actions[0] == 3:
            pos_prime[0] = pos[0] + 1
        if actions[1] == 0:
            pos_prime[1] = pos[1] - self.nRows
        elif actions[1] == 1:
            pos_prime[1] = pos[1] + self.nRows
        elif actions[1] == 2:
            pos_prime[1] = pos[1] - 1
        elif actions[1] == 3:
            pos_prime[1] = pos[1] + 1
        if grid_list[pos_prime[0]] == 1 or grid_list[pos[0]] == 3 :
            pos_prime[0] = pos[0]
        if grid_list[pos_prime[1]] == 1 or grid_list[pos[1]] == 3 :
            pos_prime[1] = pos[1]
        agents.set_position(pos_prime)
        if pos_prime[0] == pos_prime[1]:
            collision_flag = True
            if pos_prime[1] != pos[1]:
                pos_prime[1] = pos[1]
            else:
                pos_prime[0] = pos[0]
            inst_rew = -2
        s_prime = tuple_to_state(pos_prime)
        return s_prime,inst_rew, collision_flag

    def comeback(self, agent, pos):
        agent.set_position(pos)

    # method to set the environment random seed
    def _seed(self, seed):
        np.random.seed(seed)

def tuple_to_state(tuple):
        s = tuple[0]*25 + tuple[1]
        return s

def coupled_action(a):
    action_pair = [int(a/4),int(a%4)]
    return action_pair

def tuple_to_action(tuple):
    action = tuple[0]*4 + tuple[1]
    return action

# definition of the greedy policy for our model
def eps_greedy(s1, s2, Q, eps, allowed_actions):
  if np.random.rand() <= eps:
    actions1 = np.where(allowed_actions[s1])
    actions2 = np.where(allowed_actions[s2])
    actions1 = actions1[0] # just keep the indices of the allowed actions
    actions2 = actions2[0]
    a1 = np.random.choice(actions1, p=(np.ones(len(actions1)) / len(actions1)))
    a2 = np.random.choice(actions2, p=(np.ones(len(actions2)) / len(actions2)))
    a = tuple_to_action([a1,a2])
  else:
    s = tuple_to_state([s1,s2])
    actions1 = np.where(allowed_actions[s1])[0]
    actions2 = np.where(allowed_actions[s2])[0]
    Q_s = Q[s, :].copy()
    i = 0
    for qs in Q_s:
        s_prime1 = int(i/4)
        s_prime2 = int(i%4)
        if (s_prime1 not in actions1) or (s_prime2 not in actions2):
            Q_s[i] = -np.inf
        i+=1
    a = np.argmax(Q_s)
  return a

def multi_agent_qlearning(epochs, ep_length, gamma, seed):
    agents = Agents('Spiky','Roby',[0,4])
    collisions = []

    env1 = Environment()

    env1._seed(seed)

    # learning parameters
    M = epochs
    m = 1
    k = ep_length # length of the episode
    # initial Q function
    Q = np.zeros((env1.nS,env1.nA))

    # Keeps track of useful statistics
    episodes_reward = np.zeros((epochs))

    while m<M:
        if m == 199:
            print('x')
        print('iteretion n.',m)
        alpha = (1 - m/M)
        eps = (1 - m/M) ** 2
        # initial state and action
        pos = agents.get_position()
        s = tuple_to_state(pos)
        a = eps_greedy(pos[0],pos[1], Q, eps, env1.allowed_actions)
        # execute an entire episode of two actions
        for i in range(0,k):
            s_prime, reward, collision = env1.transition_model(agents,a)
            if collision:
                collisions.append(m)
                break
            if s_prime == 447 or s_prime == 567:
                break
            # policy improvement step
            a_prime = eps_greedy(agents.get_position()[0],agents.get_position()[1],Q,eps, env1.allowed_actions)
            # Update stats
            episodes_reward[m] += reward
            # Q-learning update
            Q[s, a] = Q[s, a] + alpha * (reward + gamma * np.max(Q[s_prime, :]) - Q[s, a])
            s = s_prime
            a = a_prime
        # next iteration
        m = m + 1
        env1.comeback(agents,[0,4])
        print('Q matrix updated:')
        print(Q)
        print('----------------------------------------------')
    print(collisions)
    return Q,episodes_reward

def confidency_gaps(n):
    mean = np.zeros(n)
    std = np.zeros(n)
    for i in range(0,n):
        # ogni esperimento è eseguito con seed diversi
        Q, ep_reward= multi_agent_qlearning(epochs=200, ep_length=8, gamma=0.9, seed=i)
        # ep_reward matrice 2xM dove M sono le epoche, contiene il reward totale per ogni episodio
        mean[i] = np.mean(ep_reward) # calcolo la media del reward cumulato nell'esperimento
        std[i] = np.std(ep_reward)/n # calcolo la dv. standard del reward cumulato nell'esperimento

    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(15, 5))
    ax.set_title('Agents')
    ax.plot(mean, color='blue')
    ax.fill_between(range(0,len(mean)), (mean - std), (mean + std), alpha = .3)
    ax.set_xlabel('Experiments')
    ax.set_ylabel('Rewards')
    ax.set_xticks(np.arange(0,n,1))
    plt.show()

#Q, ep_reward = multi_agent_qlearning(epochs=200,ep_length=8,gamma=0.9, seed=43)
#print(ep_reward[199])
fig, ax = plt.subplots()
#ax.plot(ep_reward)
ax.set_xlabel('episode')
ax.set_ylabel('reward')
#plt.show()

confidency_gaps(10)