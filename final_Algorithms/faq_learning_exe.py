import numpy as np
import matplotlib.pyplot as plt
import asyncio
from bleak import BleakScanner, BleakClient
import time

class Agent():

    def __init__(self, name, pos):
        self.name = name
        self.position = pos

    def set_position(self,pos):
        self.position = pos

    def get_position(self):
        return self.position

class Connection():

    def __init__(self,hub_name, service, rx, tx):
        self.hub = hub_name
        self.UART_SERVICE_UUID = service
        self.UART_RX_CHAR_UUID = rx
        self.UART_TX_CHAR_UUID = tx
        self.client = BleakClient('x')
        self.queue = asyncio.Queue()

    def hub_filter(self, device, ad):
        return device.name and device.name.lower() == self.hub.lower()

    def getName(self):
        return self.hub
    def setName(self,name):
        self.hub = name
    def getService(self):
        return self.UART_SERVICE_UUID
    def getTx(self):
        return self.UART_TX_CHAR_UUID
    def getRx(self):
        return self.UART_RX_CHAR_UUID
    def setService(self, s):
        self.UART_SERVICE_UUID = s
    def setTx(self, tx):
        self.UART_TX_CHAR_UUID = tx
    def setRx(self, rx):
        self.UART_RX_CHAR_UUID = rx
    async def getData(self):
        obj = await self.queue.get()
        return obj
    def handle_disconnect(self, _):
        print("Hub was disconnected.")

    async def handle_rx(self, _, data: bytearray):
        #print("Received:", data)
        d = str(data,'utf-8')
        print(d)
        await self.queue.put((d))

    async def connect(self):
        device = await BleakScanner.find_device_by_filter(self.hub_filter)
        self.client = BleakClient(device, disconnected_callback=self.handle_disconnect)
        # Connect and get services.
        await self.client.connect()
        await self.client.start_notify(self.UART_TX_CHAR_UUID, self.handle_rx)



    async def send_message(self,data):
        nus = self.client.services.get_service(self.UART_SERVICE_UUID)
        rx_char = nus.get_characteristic(self.UART_RX_CHAR_UUID)
        await self.client.write_gatt_char(rx_char, data)

    async def disconnect(self):
        await self.client.disconnect()
        self.client = BleakClient('x')

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

async def transition_model(env,agent1,a1,client1, agent2, a2, client2):
    # action encoding:
    # 0: up
    # 1: down
    # 2: left
    # 3: right
    grid = env.grid.flatten()
    s1 = agent1.get_position()
    s2 = agent2.get_position()
    sp1 = state_upgrade(s1,a1)
    sp2 = state_upgrade(s2,a2)
    if grid[s1] == 3:
        inst_rew1 = 1.0
        sp1 = s1
        a1 = 9
    elif grid[s2] == 3:
        inst_rew2 = 1.0
        sp2 = s2
        a2 = 9
    # for collisions we set agent1 as the master
    if sp1 == s2:
        sp1 = s1
        inst_rew1 = -1.0
    else:
        if a1 != 9:
            # ack robot 1
            await client1.send_message(b'ack1')
            time.sleep(0.2)
            str1 = str(a1)
            await client1.send_message(bytes(str1,'utf-8'))
            # wait the robots to finish actions
            await client1.getData()
            # ack robot 1
            await client1.send_message(b'ack1')
            time.sleep(0.2)
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
                time.sleep(0.2)
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
        if a2 != 9:
            # ack robot 2
            await client2.send_message(b'ack1')
            time.sleep(0.2)
            str2 = str(a2)
            await client2.send_message(bytes(str2,'utf-8'))
            # wait the robots to finish actions
            await client2.getData()
            # ack robot 1
            await client2.send_message(b'ack1')
            time.sleep(0.2)
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
                time.sleep(0.2)
                undo2 = undoMove(a2)
                str2 = str(undo2)
                await client2.send_message(bytes(str2,'utf-8'))
                await client2.getData()
            elif c2 == 'g':
                inst_rew2 = 1.0
    agent1.set_position(sp1)
    agent2.set_position(sp2)
    return sp1,sp2, inst_rew1, inst_rew2

# definition of the greedy policy for our model
def eps_greedy(s, Q, eps, allowed_actions):
    actions = np.where(allowed_actions[s])
    actions = actions[0] # just keep the indices of the allowed actions
    if np.random.rand() <= eps:
        mult = len(actions)
        a = np.random.choice(actions, p=(np.ones(len(actions)) / len(actions)))
        xi = eps/(mult-1)
    else:
        Q_s = Q[s, :].copy()
        Q_s[allowed_actions[s] == 0] = - np.inf
        a = np.argmax(Q_s)
        xi = 1 - eps
    return a, xi

async def faq_learning(epochs, ep_length, beta, gamma, seed1, seed2, eps_mode):
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
    #m = 0
    m=45
    k = ep_length # length of the episode
    # initial Q function
    #Q1 = np.zeros((env1.nS,env1.nA))
    #Q2 = np.zeros((env2.nS,env2.nA))
    Q1 = np.array([[ 0.        , -0.09779902,  0.        , -0.1       ],       [ 0.        , -0.99924375, -0.1       , -0.1       ],       [ 0.        , -0.9875    , -0.09999999, -0.51963265],       [ 0.        , -0.07555858, -0.09875   , -0.05135664],       [ 0.        , -0.05135664,  0.        ,  0.        ],       [-0.09848214, -0.09586546,  0.        , -0.608     ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [-0.05135664, -0.04975385, -0.49753846,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [-0.09995655, -0.09885698,  0.        , -0.11444287],       [-0.5859375 , -0.05859375, -0.04896763, -0.06515536],       [-0.42830966,  0.39049808,  0.        ,  0.        ],       [-0.04975385,  0.        ,  0.        ,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [-0.09886965, -0.09233538,  0.        , -0.09017723],       [-0.0530303 , -0.36788281, -0.03377757,  0.25512057],       [ 0.50373248,  0.        ,  0.        ,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [-0.09233538,  0.        ,  0.        , -0.48188235],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ]])
    Q2 = np.array([[ 0.        , -0.05708571,  0.        ,  0.        ],       [ 0.        , -0.57085714, -0.05708571, -0.0787623 ],       [ 0.        , -0.9945    , -0.58104049, -0.09966824],       [ 0.        , -0.12316147, -0.51948759, -0.1       ],       [ 0.        , -0.02322563, -0.1       ,  0.        ],       [-0.05708571,  0.        ,  0.        ,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [-0.0891208 , -0.09998438, -0.9875    , -0.08020005],       [-0.09999984, -0.09843474, -0.13486734,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [-0.53030303,  0.53030303,  0.        ,  0.        ],       [-0.09998438, -0.53030303, -0.0530303 , -0.05135664],       [-0.25901003, -0.09999999, -0.54328443,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [-0.09988247, -0.07046208, -0.54857143,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [ 0.        ,  0.        ,  0.        ,  0.        ],       [ 0.86342688,  0.        ,  0.        ,  0.        ],       [-0.25512057,  0.        ,  0.95627498,  0.        ],       [-0.09086782,  0.        ,  0.22702114,  0.        ]])
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
            a1,xi1 = eps_greedy(s1, Q1, eps, env1.allowed_actions)
            a2,xi2 = eps_greedy(s2, Q2, eps, env2.allowed_actions)
            # let the robot go out the dock
            # ack robot 1
            await robot1.send_message(b'ack1')
            time.sleep(0.2)
            str1 = str(5)
            await robot1.send_message(bytes(str1,'utf-8'))
            # ack robot 2
            await robot2.send_message(b'ack2')
            time.sleep(0.2)
            str2 = str(5)
            await robot2.send_message(bytes(str2,'utf-8'))
            # wait the robots to finish actions
            await robot1.getData()
            await robot2.getData()
            # execute an entire episode of k actions
            for i in range(0,k):
                print('Actions: ', a1,a2)
                s_prime1,s_prime2, reward1, reward2 = await transition_model(env1,spiky,a1, robot1, roby, a2, robot2)
                # Q-learning update
                Q1[s1, a1] = Q1[s1, a1] + np.min([beta/xi1,1]) * alpha * (reward1 + gamma * np.max(Q1[s_prime1, :]) - Q1[s1, a1])
                Q2[s2, a2] = Q2[s2, a2] + np.min([beta/xi2,1]) * alpha * (reward2 + gamma * np.max(Q2[s_prime2, :]) - Q2[s2, a2])
                # policy improvement step
                a_prime1,xi1 = eps_greedy(s_prime1,Q1,eps, env1.allowed_actions)
                a_prime2,xi2 = eps_greedy(s_prime2,Q2,eps, env2.allowed_actions)
                s1 = s_prime1
                a1 = a_prime1
                s2 = s_prime2
                a2 = a_prime2
                if (s1 == 17 or s1 == 22) and (s2 == 17 or s2 == 22):
                    break
            print('epochs ', m)
            # next iteration
            m = m + 1
            print('Comeback')
            # ack robot 1
            await robot1.send_message(b'ack1')
            time.sleep(0.2)
            str1 = str(7)
            await robot1.send_message(bytes(str1,'utf-8'))

            env1.comeback(spiky,0)
            # ack robot 2
            await robot2.send_message(b'ack2')
            time.sleep(0.2)
            str2 = str(7)
            await robot2.send_message(bytes(str2,'utf-8'))
            env2.comeback(roby,4)

            await robot1.getData()
            await robot2.getData()
            #back in the hub
            # ack robot 1
            await robot1.send_message(b'ack1')
            time.sleep(0.2)
            str1 = str(6)
            await robot1.send_message(bytes(str1,'utf-8'))

            # ack robot 2
            await robot2.send_message(b'ack2')
            time.sleep(0.2)
            str2 = str(6)
            await robot2.send_message(bytes(str2,'utf-8'))

            await robot1.getData()
            await robot2.getData()
        # ack robot 1
        await robot1.send_message(b'ack1')
        time.sleep(0.2)
        str1 = str(9)
        await robot1.send_message(bytes(str1,'utf-8'))
        # ack robot 2
        await robot2.send_message(b'ack2')
        time.sleep(0.2)
        str2 = str(9)
        await robot2.send_message(bytes(str2,'utf-8'))
    except Exception as e:
        print(e)
    finally:
        await robot1.disconnect()
        await robot2.disconnect()
    return Q1,Q2

asyncio.run(faq_learning(80,7,0.6,0.9,1,101,'quadratic'))