import numpy as np
import matplotlib.pyplot as plt
import asyncio
from bleak import BleakScanner, BleakClient
from centralized_learning.qlearning_centralized import Connection
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

def state_upgrade(s,a):
    if a == 0:
        s_prime = s - 5
    elif a == 1:
        s_prime = s + 5
    elif a == 2:
        s_prime = s - 1
    elif a == 3:
        s_prime = s + 1
    return s_prime

def undoMove(a):
    if a == 0:
        u = 1
    elif a == 1:
        u = 0
    elif a == 2:
        u = 3
    elif a == 3:
        u = 2
    return u

def tuple_to_state(tuple):
        s = tuple[0]*25 + tuple[1]
        return s

def coupled_action(a):
    action_pair = [int(a/4),int(a%4)]
    return action_pair

def tuple_to_action(tuple):
    action = tuple[0]*4 + tuple[1]
    return action

async def transition_model(env,agent1,client1, agent2, client2, a):
    # action encoding:
    # 0: up
    # 1: down
    # 2: left
    # 3: right
    s1 = agent1.get_position()
    s2 = agent2.get_position()
    a1,a2 = coupled_action(a)
    grid = env.grid.flatten()
    if grid[s1] == 3:
        inst_rew1 = 1.0
        sp1 = s1
    elif grid[s2] == 3:
        inst_rew2 = 1.0
        sp2 = s2
    else:
        sp1 = state_upgrade(s1,a1)
        sp2 = state_upgrade(s2,a2)
        # for collisions we set agent1 as the master
        if sp1 == s2:
            sp1 = s1
            inst_rew1 = -1.0
        else:
            # ack robot 1
            await client1.send_message(b'ack1')
            time.sleep(0.5)
            str1 = str(a1)
            await client1.send_message(bytes(str1,'utf-8'))
            # wait the robots to finish actions
            await client1.getData()
            # ack robot 1
            await client1.send_message(b'ack1')
            time.sleep(0.5)
            #check color
            str1 = str(4)
            await client1.send_message(bytes(str1,'utf-8'))
            c1 = await client1.getData()
            await client1.getData()
            if c1 == 'b':
                inst_rew1 = -0.1
            elif c1 == 'r':
                inst_rew1 = -1.0
                sp1 = s1
                # ack robot 1
                await client1.send_message(b'ack1')
                time.sleep(0.5)
                undo1 = undoMove(a1)
                str1 = str(undo1)
                await client1.send_message(bytes(str1,'utf-8'))
                await client1.getData()
            elif c1 == 'g':
                inst_rew1 = 1.0
        if sp2 == sp1:
            sp2 = s2
            inst_rew2 = -1.0
        else:
            # ack robot 1
            await client2.send_message(b'ack1')
            time.sleep(0.5)
            str2 = str(a2)
            await client2.send_message(bytes(str2,'utf-8'))
            # wait the robots to finish actions
            await client2.getData()
            # ack robot 1
            await client2.send_message(b'ack1')
            time.sleep(0.5)
            #check color
            str2 = str(4)
            await client2.send_message(bytes(str2,'utf-8'))
            c2 = await client2.getData()
            await client2.getData()
            if c2 == 'b':
                inst_rew2 = -0.1
            elif c2 == 'r':
                inst_rew2 = -1.0
                sp2 = s2
                # ack robot 1
                await client2.send_message(b'ack1')
                time.sleep(0.5)
                undo2 = undoMove(a2)
                str2 = str(undo2)
                await client2.send_message(bytes(str2,'utf-8'))
                await client2.getData()
            elif c2 == 'g':
                inst_rew2 = 1.0
        agent1.set_position(sp1)
        agent2.set_position(sp2)
        inst_rew = np.min([inst_rew1,inst_rew2])
        return sp1,sp2, inst_rew

# definition of the greedy policy for our model
def eps_greedy(s1, s2, Q, eps, allowed_actions):
    actions1 = np.where(allowed_actions[s1])
    actions2 = np.where(allowed_actions[s2])
    actions1 = actions1[0] # just keep the indices of the allowed actions
    actions2 = actions2[0]
    if np.random.rand() <= eps:
        a1 = np.random.choice(actions1, p=(np.ones(len(actions1)) / len(actions1)))
        a2 = np.random.choice(actions2, p=(np.ones(len(actions2)) / len(actions2)))
        a = tuple_to_action([a1,a2])
    else:
        s = tuple_to_state([s1,s2])
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

async def centralized_qlearning(epochs, ep_length, gamma, seed, eps_mode):
    spiky = Agent('Spiky',0)
    roby = Agent('Roby',4)
    env = Environment()
    # they generare the allowed actions for the grid and the reward distribution
    env.setup()
    # randomize the experiment
    env._seed(seed)
    # learning parameters
    M = epochs
    m = 0
    k = ep_length # length of the episode
    # initial Q function
    Q = np.zeros((env.nS**2,env.nA**2))
    # generate the two connections
    robot1 = Connection('Spiky',"6E400001-B5A3-F393-E0A9-E50E24DCCA9E","6E400002-B5A3-F393-E0A9-E50E24DCCA9E", "6E400003-B5A3-F393-E0A9-E50E24DCCA9E")
    robot2 = Connection('Roby',"6E400001-B5A3-F393-E0A9-E50E24DCCA9E","6E400002-B5A3-F393-E0A9-E50E24DCCA9E", "6E400003-B5A3-F393-E0A9-E50E24DCCA9E")

    try:
        await robot1.connect()
        await robot2.connect()
        print('click the robots to start')
    # wait the first ack from the robots
        await robot1.getData()
        await robot2.getData()

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
            # initial state and action
            s1 = spiky.get_position()
            s2 = roby.get_position()
            s = tuple_to_state([s1,s2])
            a = eps_greedy(s1,s2, Q, eps, env.allowed_actions)
            # ack robot 1
            await robot1.send_message(b'ack1')
            time.sleep(0.5)
            str1 = str(5)
            await robot1.send_message(bytes(str1,'utf-8'))
            # ack robot 2
            await robot2.send_message(b'ack2')
            time.sleep(0.5)
            str2 = str(5)
            await robot2.send_message(bytes(str2,'utf-8'))
            # wait the robots to finish actions
            await robot1.getData()
            await robot2.getData()
            # execute an entire episode of k actions
            for i in range(0,k):
                s1_prime,s2_prime, reward = await transition_model(env,spiky, robot1, roby, robot2, a)
                # Q-learning update
                s_prime = tuple_to_state([s1_prime,s2_prime])
                Q[s, a] = Q[s, a] + alpha * (reward + gamma * np.max(Q[s_prime, :]) - Q[s, a])
                # policy improvement step
                a_prime = eps_greedy(s1_prime,s2_prime,Q,eps, env.allowed_actions)
                s = s_prime
                a = a_prime
                if (s1 == 17 or s1 == 22) and (s2 == 17 or s2 == 22):
                    break
            print('epochs ', m)
            # next iteration
            m = m + 1
            # ack robot 1
            await robot1.send_message(b'ack1')
            time.sleep(0.5)
            str1 = str(7)
            await robot1.send_message(bytes(str1,'utf-8'))
            await robot1.getData()
            env.comeback(spiky,0)
            # ack robot 2
            await robot2.send_message(b'ack2')
            time.sleep(0.5)
            str2 = str(7)
            await robot2.send_message(bytes(str2,'utf-8'))
            await robot2.getData()
            env.comeback(roby,4)
        # ack robot 1
        await robot1.send_message(b'ack1')
        time.sleep(0.5)
        str1 = str(9)
        await robot1.send_message(bytes(str1,'utf-8'))
        # ack robot 2
        await robot2.send_message(b'ack2')
        time.sleep(0.5)
        str2 = str(9)
        await robot2.send_message(bytes(str2,'utf-8'))
    except Exception as e:
        print(e)
    finally:
        await robot1.disconnect()
        await robot2.disconnect()
    return Q

asyncio.run(centralized_qlearning(400,8,0.9,10,'cubic'))