import asyncio
from bleak import BleakScanner, BleakClient
import time
import numpy as np


# centralized algorithm to handle a qlearning path on two different robot

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
        self.nS = 25
        self.nA = 2
        self.mu = [0,0,1,0,0]
        self.gamma = 0.9
        self.currentStates = [2,2]

    # method to set the environment random seed
    def _seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.currentStates = [2,2]
        s = self.tuple_to_state(self.currentStates)
        return s

    def tuple_to_state(self,tuple):
        s = tuple[0]*5 + tuple[1]
        return s

    def transition_model(self,a):
        cs = self.currentStates
        if a[0] == 0 and a[1] == 0:
            self.currentStates = [cs[0]+1,cs[1]+1]
            s_prime = self.currentStates
            reward = -1
        elif a[0] == 1 and a[1] == 1:
            self.currentStates = [cs[0]-1,cs[1]-1]
            s_prime = self.currentStates
            reward = -1
        elif a[0] == 1 and a[1] == 0:
            self.currentStates = [cs[0]-1,cs[1]+1]
            s_prime = self.currentStates
            if (s_prime[0] == 0 and s_prime[1] == 4) or (s_prime[0] == 4 and s_prime[1] == 0):
                reward = 1
            else:
                reward = -0.1
        elif a[0] == 0 and a[1] == 1:
            self.currentStates = [cs[0]+1,cs[1]-1]
            s_prime = self.currentStates
            if (s_prime[0] == 0 and s_prime[1] == 4) or (s_prime[0] == 4 and s_prime[1] == 0):
                reward = 1
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
        Q_s = Q[s,:].copy()
        a = np.argmax(Q_s)
    return a

async def comeback(pos,r1,r2):
    one = pos[0]
    if one >= 2:
        iterations = one - 2
        for i in range(0,iterations):
            r1.send_message(b'ack1')
            time.sleep(0.5)
            r1.send_message(b'1')
    else:
        iterations = 2-one
        for i in range(0,iterations):
            r1.send_message(b'ack1')
            time.sleep(0.5)
            r1.send_message(b'0')
    two = pos[1]
    if two >= 2:
        iterations = two - 2
        for i in range(0,iterations):
            r2.send_message(b'ack2')
            time.sleep(0.5)
            r2.send_message(b'1')
    else:
        iterations = 2-two
        for i in range(0,iterations):
            r2.send_message(b'ack2')
            time.sleep(0.5)
            r2.send_message(b'0')

async def qLearning():
    env = Environment()
    env._seed(10)
    # learning parameters
    M = 30
    m = 1
    k = 2 # length of the episode
    # initial Q function
    Q = np.zeros((env.nS,env.nA**2))

    # generate the two connections
    robot1 = Connection('Spiky',"6E400001-B5A3-F393-E0A9-E50E24DCCA9E","6E400002-B5A3-F393-E0A9-E50E24DCCA9E", "6E400003-B5A3-F393-E0A9-E50E24DCCA9E")
    robot2 = Connection('Roby',"6E400001-B5A3-F393-E0A9-E50E24DCCA9E","6E400002-B5A3-F393-E0A9-E50E24DCCA9E", "6E400003-B5A3-F393-E0A9-E50E24DCCA9E")
    try:
        await robot1.connect()
        await robot2.connect()
        print('click the robots to start')
        await robot1.getData()
        await robot2.getData()
        # waiting for the starting function
        #print('robots need to be initiated')
        #await robot1.getData()
        #await robot2.getData()
        print('Starting...')
        time.sleep(1)
        while m<M:
            alpha = (1 - m/M)
            eps = (1 - m/M) ** 3
            # initial state and action
            s = env.reset()
            a = eps_greedy(s, Q, eps)
            # execute an entire episode of two actions
            for i in range(0,k):
                actions = action_to_pair(a)
                # send actions to robots
                # ack robot 1
                await robot1.send_message(b'ack1')
                time.sleep(0.5)
                str1 = str(actions[0])
                await robot1.send_message(bytes(str1,'utf-8'))

                # ack robot 2
                await robot2.send_message(b'ack2')
                time.sleep(0.5)
                str2 = str(actions[1])
                await robot2.send_message(bytes(str2,'utf-8'))

                # wait the robots to finish actions
                await robot1.getData()
                await robot2.getData()

                s_prime, reward = env.transition_model(actions)
                # policy improvement step
                a_prime = eps_greedy(s_prime,Q,eps)
                # Q-learning update
                Q[s, a] = Q[s, a] + alpha * (reward + env.gamma * np.max(Q[s_prime, :]) - Q[s, a])
                s = s_prime
                a = a_prime
            # next iteration
            print('iteretion n.',m)
            m = m + 1
            print('Q matrix updated:')
            print(Q)
            print('----------------------------------------------')
            # wait the robots to finish the comeback
            await robot1.getData()
            await robot2.getData()


        await robot1.send_message(b'9')
        await robot2.send_message(b'9')
        return Q

    except Exception as e:
        print(e)
    finally:
        await robot1.disconnect()
        await robot2.disconnect()


asyncio.run(qLearning())