import numpy as np

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
            self.nS = self.nRows*self.nCols**2
            self.nA = 4**2
            self.nO = 4 # number of ostacle on the grid
            self.gamma = 0.9
            self.grid = np.array([[2,0,0,0,2],
                         [0,1,1,0,0],
                         [0,0,0,0,0],
                         [0,0,3,1,0],
                         [0,1,3,0,0]])

    def tuple_to_state(tuple):
        s = tuple[0]*25 + tuple[1]
        return s

    def coupled_action(a):
        action_pair = [int(a/4),int(a%4)]
        return action_pair


    def transition_model(self,agent,a):

        # action encoding:
        # 0: up
        # 1: down
        # 2: left
        # 3: right
        pos = agent.get_position()
        s = self.tuple_to_state(pos)
        actions = self.coupled_action(a)
        pos_prime = [0,0]
        grid_list = self.grid.flatten()
        if actions[0] == 0:
            if (pos[0] - self.nRows) >= 0:
                pos_prime[0] = pos[0] - self.nRows

        elif actions[0] == 1:
            s_prime = pos[0] + self.nRows
            agent.set_position(s_prime)
        elif actions[0] == 2:
            s_prime = pos[0] - 1
            agent.set_position(s_prime)
        elif actions[0] == 3:
            s_prime = pos[0] + 1
            agent.set_position(s_prime)
        grid_array = self.grid.flatten()
        if grid_array[s_prime] == 1 or grid_array[s] == 3 :
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
def eps_greedy(s, Q, eps):
  if np.random.rand() <= eps:
    actions = np.where(allowed_actions[s])
    actions = actions[0] # just keep the indices of the allowed actions
    a = np.random.choice(actions, p=(np.ones(len(actions)) / len(actions)))
  else:
    Q_s = Q[s, :].copy()
    Q_s[allowed_actions[s] == 0] = - np.inf
    a = np.argmax(Q_s)
  return a