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
            if grid_array[s_prime] == 1 or grid_array[s] == 3:
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

async def check_color(robot, a, color):
    if color == 'b':
        reward = -0.1
    elif color == 'r':
        reward = -1.0
        # ack robot 1
        await robot.send_message(b'ack1')
        undo2 = undoMove(a)
        str2 = str(undo2)
        await robot.send_message(bytes(str2,'utf-8'))
        await robot.getData()
    elif color == 'g':
        reward = 1.0
    return reward

def collision_on_red(s1,s2,a1,a2):
    if a1 == 0:
        s_prime1 = s1 - 5
    elif a1 == 1:
        s_prime1 = s1 + 5
    elif a1 == 2:
        s_prime1 = s1 - 1
    elif a1 == 3:
        s_prime1 = s1 + 1
    if a2 == 0:
        s_prime2 = s2 - 5
    elif a2 == 1:
        s_prime2 = s2 + 5
    elif a2 == 2:
        s_prime2 = s2 - 1
    elif a2 == 3:
        s_prime2 = s2 + 1
    if s_prime1 != s_prime2:
        return True
    else:
        return False

def update_function(Q, alpha, gamma, s, s_prime, a, r, allowed_actions,beta,xi):
    Q_sp = Q[s_prime,:].copy()
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
    Q[s,a] = Q[s,a] + alpha * np.min([beta/xi,1]) (r + (gamma * np.max(Q_sp)) - Q[s,a])
    return Q

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
    m = 0
    #m=45
    k = ep_length # length of the episode
    # initial Q function
    Q1 = np.zeros((env1.nS,env1.nA))
    Q2 = np.zeros((env2.nS,env2.nA))
    #Q1 = np.load('/Users/alessandrodonofrio/Desktop/Spike Prime Python/final_Algorithms/faqQ1.npy')
    #Q2 = np.load('/Users/alessandrodonofrio/Desktop/Spike Prime Python/final_Algorithms/faqQ2.npy')

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
                s1 = spiky.get_position()
                s2 = roby.get_position()
                print('Actions: ', a1,a2,'States: ', s1,s2)
                s_prime1, target1 = env1.transition_model(spiky,a1)
                s_prime2, target2 = env2.transition_model(roby,a2)
                # collision control
                collision_flag1 = False
                collision_flag2 = False
                # case 1: robots reach the same state
                if s_prime1 == s_prime2:
                    r = -1.0
                    # agent 1 is the master, so agent2 gets the bad reward
                    grid_flatten = env2.grid.flatten()
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
                # case 2: robots try to exchange the state
                if (s1 == s_prime2 and s2 == s_prime1):
                    r = -1.0
                    collision_flag1 = True
                    collision_flag2 = True
                    s_prime1 = s1
                    spiky.set_position(s_prime1)
                    reward1 = r
                    s_prime2 = s2
                    roby.set_position(s_prime2)
                    reward2 = r
                #moving the hubs
                # case when no collisions are possible
                if (collision_flag1 == False and collision_flag2 == False) and (target1 == False and target2 == False) and (s_prime2 != s1) and (s_prime1 != s2):
                    # case 1: robots have not chosen to go on the same obstacle
                    if collision_on_red(s1,s2,a1,a2):
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

                        # check colors
                        # ack robot 1
                        await robot1.send_message(b'ack1')
                        str1 = str(4)
                        await robot1.send_message(bytes(str1,'utf-8'))
                        c1 = await robot1.getData()
                        await robot1.getData()
                        reward1 = await check_color(robot1, a1, c1)
                        # ack robot 2
                        await robot2.send_message(b'ack2')
                        str2 = str(4)
                        await robot2.send_message(bytes(str2,'utf-8'))
                        c2 = await robot2.getData()
                        await robot2.getData()
                        reward2 = await check_color(robot2,a2,c2)
                    # case 2: the opposite of case 1, it handles the movement one at a time
                    else:
                        # ack robot 1
                        await robot1.send_message(b'ack1')
                        str1 = str(a1)
                        await robot1.send_message(bytes(str1,'utf-8'))
                        # wait the robots to finish actions
                        await robot1.getData()
                        # check color
                        # ack robot 1
                        await robot1.send_message(b'ack1')
                        str1 = str(4)
                        await robot1.send_message(bytes(str1,'utf-8'))
                        c1 = await robot1.getData()
                        await robot1.getData()
                        reward1 = await check_color(robot1, a1, c1)

                        # ack robot 2
                        await robot2.send_message(b'ack2')
                        str2 = str(a2)
                        await robot2.send_message(bytes(str2,'utf-8'))
                        # wait the robots to finish actions
                        await robot2.getData()
                        # check color
                        # ack robot 2
                        await robot2.send_message(b'ack2')
                        str2 = str(4)
                        await robot2.send_message(bytes(str2,'utf-8'))
                        c2 = await robot2.getData()
                        await robot2.getData()
                        reward2 = await check_color(robot2,a2,c2)
                # case 3: robot 1 goes in the state of robot 2, robot 2 moves first
                elif s_prime1 == s2 and (collision_flag1 == False and collision_flag2 == False):
                    # ack robot 2
                    await robot2.send_message(b'ack2')
                    str2 = str(a2)
                    await robot2.send_message(bytes(str2,'utf-8'))
                    # wait the robots to finish actions
                    await robot2.getData()
                    # check color
                    # ack robot 2
                    await robot2.send_message(b'ack2')
                    str2 = str(4)
                    await robot2.send_message(bytes(str2,'utf-8'))
                    c2 = await robot2.getData()
                    await robot2.getData()
                    reward2 = await check_color(robot2,a2,c2)

                    # ack robot 1
                    await robot1.send_message(b'ack1')
                    str1 = str(a1)
                    await robot1.send_message(bytes(str1,'utf-8'))
                    # wait the robots to finish actions
                    await robot1.getData()
                    # check color
                    # ack robot 1
                    await robot1.send_message(b'ack1')
                    str1 = str(4)
                    await robot1.send_message(bytes(str1,'utf-8'))
                    c1 = await robot1.getData()
                    await robot1.getData()
                    reward1 = await check_color(robot1, a1, c1)
                # case 4: robot 2 goes in the state of robot 1, robot 1 moves first
                elif s_prime2 == s1 and (collision_flag1 == False and collision_flag2 == False):
                    # ack robot 1
                    await robot1.send_message(b'ack1')
                    str1 = str(a1)
                    await robot1.send_message(bytes(str1,'utf-8'))
                    # wait the robots to finish actions
                    await robot1.getData()
                    #check color
                    # ack robot 1
                    await robot1.send_message(b'ack1')
                    str1 = str(4)
                    await robot1.send_message(bytes(str1,'utf-8'))
                    c1 = await robot1.getData()
                    await robot1.getData()
                    reward1 = await check_color(robot1, a1, c1)

                    # ack robot 2
                    await robot2.send_message(b'ack2')
                    str2 = str(a2)
                    await robot2.send_message(bytes(str2,'utf-8'))
                    # wait the robots to finish actions
                    await robot2.getData()
                    # check color
                    # ack robot 2
                    await robot2.send_message(b'ack2')
                    str2 = str(4)
                    await robot2.send_message(bytes(str2,'utf-8'))
                    c2 = await robot2.getData()
                    await robot2.getData()
                    reward2 = await check_color(robot2,a2,c2)
                else:  # collisions cases
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
                        reward1 = await check_color(robot1, a1, c1)
                    if collision_flag2 != True and target2 == False:
                        # ack robot 2
                        await robot2.send_message(b'ack1')
                        str2 = str(a2)
                        await robot2.send_message(bytes(str2,'utf-8'))
                        # wait the robots to finish actions
                        await robot2.getData()
                        # ack robot 2
                        await robot2.send_message(b'ack1')
                        #check color
                        str2 = str(4)
                        await robot2.send_message(bytes(str2,'utf-8'))
                        c2 = await robot2.getData()
                        await robot2.getData()
                        reward2 = await check_color(robot2,a2,c2)
                # Q-learning update
                Q1 = update_function(Q1,alpha,gamma,s1,s_prime1,a1,reward1, env1.allowed_actions, beta, xi1)
                Q2 = update_function(Q2,alpha,gamma,s2,s_prime2,a2,reward2, env2.allowed_actions, beta, xi2)
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

            # save the matrix in a file
            np.save('/Users/alessandrodonofrio/Desktop/Spike Prime Python/final_Algorithms/faqQ1.npy', Q1)
            np.save('/Users/alessandrodonofrio/Desktop/Spike Prime Python/final_Algorithms/faqQ2.npy', Q2)
            if (m-1)%20 == 0:
                path = '/Users/alessandrodonofrio/Desktop/Spike Prime Python/final_Algorithms/'
                file1 = 'faqQ1' + str(m-1) + '.npy'
                file2 = 'faqQ2' + str(m-1) + '.npy'
                complete_path1 = path+file1
                complete_path2 = path+file2
                f1 = open(complete_path1, 'a')
                np.save(complete_path1, Q1)
                f2 = open(complete_path2, 'a')
                np.save(complete_path2, Q2)
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
