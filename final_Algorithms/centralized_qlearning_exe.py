import sys
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

    def transition_model(self,agent,a):
    # action encoding:
        # 0: up
        # 1: down
        # 2: left
        # 3: right
        target = False
        s = agent.get_position()
        grid_array = self.grid.flatten()
        if grid_array[s] == 3:
            s_prime = s
            target = True
            return s_prime, target
        else:
            if a == 0:
                s_prime = s - self.nRows
                if s_prime < 0:
                    s_prime = s
            elif a == 1:
                s_prime = s + self.nRows
                if s_prime > 24:
                    s_prime = s
            elif a == 2:
                s_prime = s - 1
                if s_prime < 0:
                    s_prime = s
            elif a == 3:
                s_prime = s + 1
                if s_prime > 24:
                    s_prime = s
            if grid_array[s_prime] == 1:
                s_prime = s
            agent.set_position(s_prime)
            return s_prime, target

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
    #Q = np.array()
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
            str1 = str(5)
            await robot1.send_message(bytes(str1,'utf-8'))
            # ack robot 2
            await robot2.send_message(b'ack2')
            str2 = str(5)
            await robot2.send_message(bytes(str2,'utf-8'))
            # wait the robots to finish actions
            await robot1.getData()
            await robot2.getData()
            # execute an entire episode of k actions
            for i in range(0,k):
                a1,a2 = coupled_action(a)
                s1 = spiky.get_position()
                s2 = roby.get_position()
                print('Actions: ', a1,a2,'States: ', s1,s2)

                s_prime1, target1 = env.transition_model(spiky,a1)
                s_prime2, target2 = env.transition_model(roby,a2)
                # collision control
                collision_flag1 = False
                collision_flag2 = False
                if s_prime1 == s_prime2 or (s1 == s_prime2 and s2 == s_prime1):
                    r = -1.0
                    # agent 1 is the master, so agent2 gets the bad reward
                    grid_flatten = env.grid.flatten()
                    if grid_flatten[s2] != 3:
                        if s2 != s_prime2:
                            s_prime2 = s2
                            roby.set_position(s_prime2)
                            reward2 = r
                            collision_flag2 = True
                        else:
                            s_prime1 = s1
                            spiky.set_position(s_prime1)
                            reward1 = r
                            collision_flag1 = True
                    else:
                        s_prime1 = s1
                        spiky.set_position(s_prime1)
                        reward1 = r
                        collision_flag1 = True
                #moving the hubs
                if (collision_flag1 == False and collision_flag2 == False) and (target1 == False and target2 == False) and (s_prime2 != s1):
                    # ack robot 1
                    await robot1.send_message(b'ack1')
                    str1 = str(a1)
                    await robot1.send_message(bytes(str1,'utf-8'))
                    # ack robot 2
                    await robot2.send_message(b'ack2')
                    str2 = str(a2)
                    await robot2.send_message(bytes(str2,'utf-8'))
                    # wait the robots to finish actions
                    await robot1.getData()
                    await robot2.getData()
                    # ack robot 1
                    await robot1.send_message(b'ack1')
                    #check color
                    str1 = str(4)
                    await robot1.send_message(bytes(str1,'utf-8'))
                    c1 = await robot1.getData()
                    await robot1.getData()
                    if c1 == 'b':
                        reward1 = -0.1
                    elif c1 == 'r':
                        reward1 = -1.0
                        s_prime1 = s1
                        spiky.set_position(s_prime1)
                        # ack robot 1
                        await robot1.send_message(b'ack1')
                        undo1 = undoMove(a1)
                        str1 = str(undo1)
                        await robot1.send_message(bytes(str1,'utf-8'))
                        await robot1.getData()
                    elif c1 == 'g':
                        reward1 = 1.0
                    # ack robot 1
                    await robot2.send_message(b'ack1')
                    #check color
                    str2 = str(4)
                    await robot2.send_message(bytes(str2,'utf-8'))
                    c1 = await robot2.getData()
                    await robot2.getData()
                    if c1 == 'b':
                        reward2 = -0.1
                    elif c1 == 'r':
                        reward2 = -1.0
                        # ack robot 1
                        await robot2.send_message(b'ack1')
                        undo2 = undoMove(a2)
                        str2 = str(undo2)
                        await robot2.send_message(bytes(str2,'utf-8'))
                        await robot2.getData()
                    elif c1 == 'g':
                        reward2 = 1.0
                else:
                    if collision_flag1 != True and target1 == False:
                        # ack robot 1
                        await robot1.send_message(b'ack1')
                        str1 = str(a1)
                        await robot1.send_message(bytes(str1,'utf-8'))
                        # wait the robots to finish actions
                        await robot1.getData()
                        # ack robot 1
                        await robot1.send_message(b'ack1')
                        #check color
                        str1 = str(4)
                        await robot1.send_message(bytes(str1,'utf-8'))
                        c1 = await robot1.getData()
                        await robot1.getData()
                        if c1 == 'b':
                            reward1 = -0.1
                        elif c1 == 'r':
                            reward1 = -1.0
                            # ack robot 1
                            await robot1.send_message(b'ack1')
                            undo1 = undoMove(a1)
                            str1 = str(undo1)
                            await robot1.send_message(bytes(str1,'utf-8'))
                            await robot1.getData()
                        elif c1 == 'g':
                            reward1 = 1.0
                    if collision_flag2 != True and target2 == False:
                        # ack robot 1
                        await robot2.send_message(b'ack1')
                        str2 = str(a2)
                        await robot2.send_message(bytes(str2,'utf-8'))
                        # wait the robots to finish actions
                        await robot2.getData()
                        # ack robot 1
                        await robot2.send_message(b'ack1')
                        #check color
                        str2 = str(4)
                        await robot2.send_message(bytes(str2,'utf-8'))
                        c1 = await robot2.getData()
                        await robot2.getData()
                        if c1 == 'b':
                            reward2 = -0.1
                        elif c1 == 'r':
                            reward2 = -1.0
                            # ack robot 1
                            await robot2.send_message(b'ack1')
                            undo2 = undoMove(a2)
                            str2 = str(undo2)
                            await robot2.send_message(bytes(str2,'utf-8'))
                            await robot2.getData()
                        elif c1 == 'g':
                            reward2 = 1.0
                # Q-learning update
                reward = np.min([reward1,reward2])
                s_prime = tuple_to_state([s_prime1,s_prime2])
                Q = update_function(Q,alpha,gamma,s,s_prime,a,reward, env.allowed_actions)
                # policy improvement step
                a_prime = eps_greedy(s_prime1,s_prime2,Q,eps, env.allowed_actions)
                s = s_prime
                a = a_prime
                if (s1 == 17 or s1 == 22) and (s2 == 17 or s2 == 22):
                    break
            stri = 'epochs: ' + str(m)
            with open("./qValues.txt", "w") as values:
                values.write(stri)
                sQ = np.array2string(Q)
                values.write(sQ)
                values.close()
            print(stri)
            # next iteration
            m = m + 1
            print('Comeback')
            # ack robot 1
            await robot1.send_message(b'ack1')
            str1 = str(7)
            await robot1.send_message(bytes(str1,'utf-8'))

            env.comeback(spiky,0)
            # ack robot 2
            await robot2.send_message(b'ack2')
            str2 = str(7)
            await robot2.send_message(bytes(str2,'utf-8'))
            env.comeback(roby,4)

            await robot1.getData()
            await robot2.getData()
            #back in the hub
            # ack robot 1
            await robot1.send_message(b'ack1')
            str1 = str(6)
            await robot1.send_message(bytes(str1,'utf-8'))

            # ack robot 2
            await robot2.send_message(b'ack2')
            str2 = str(6)
            await robot2.send_message(bytes(str2,'utf-8'))

            await robot1.getData()
            await robot2.getData()
        # ack robot 1
        await robot1.send_message(b'ack1')
        str1 = str(9)
        await robot1.send_message(bytes(str1,'utf-8'))
        # ack robot 2
        await robot2.send_message(b'ack2')
        str2 = str(9)
        await robot2.send_message(bytes(str2,'utf-8'))
    except Exception as e:
        print(e)
    finally:
        await robot1.disconnect()
        await robot2.disconnect()
    return Q

asyncio.run(centralized_qlearning(400,8,0.9,10,'cubic'))