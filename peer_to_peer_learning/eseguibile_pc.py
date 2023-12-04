import asyncio
from bleak import BleakScanner, BleakClient
import time
import numpy as np

# environment

class Environment():
    def __init__(self):
        self.n_actions = 3
        self.n_states = 6
        self.mu = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.gamma = 0.9

    def list_states(self):
            for i in range(0,self.n_states):
                print('State '+str(i)+': The robot is '+ str(40-i*10) + ' cm distant from the obstacle.' )

    def list_actions(self):
        for i in range(0,self.n_actions):
            print('Action '+str(i) + ': the robot moves foreward of '+str((i+1)*10)+' cm.')

    # method to set the environment random seed
    def _seed(self,seed):
        np.random.seed(seed)

    # method to reset the environment to the initial state
    def reset(self):
        s = 0
        return s

# utility functions

def encode_matrix(Q):
        nr = len(Q)
        nc = len(Q[0])
        v = []
        for i in range(0,nr):
            for j in range(0,nc):
                v.append(Q[i][j])
        v_str = ''
        for el in v:
            els = str(el)
            v_str = v_str + els + '/'
        nrs = str(len(Q))
        ncs = str(len(Q[0]))
        msg = nrs + '|' + ncs + '|' + v_str
        l = len(msg)
        resto = 5 - l%5
        for x in range(0,resto):
            msg = msg + '*'
        # write the encoding function from computer to bytearray
        # Q has to be in matrix format
        byte_msg = bytes(msg,'utf-8')
        return byte_msg

def decode_matrix(msg):
    msg_split = msg.rsplit('|')
    nr = int(msg_split[0])
    nc = int(msg_split[1])
    v_string = msg_split[2]
    v = v_string.rsplit('/')
    Q = []
    for i in range(0,nr):
        q = []
        for j in range(0,nc):
            q.append(v[j+i*nc])
        Q.append(q)
    return Q

def buffer(byte_msg):
        l = len(byte_msg)
        parts = int(l/95) + 1
        buf = []
        x = str(byte_msg,'utf-8')
        if parts > 1:
            for i in range(0,parts):
                if l > (i+1)*95:
                    temp = x[i*95:(i+1)*95] + '*****'
                else:
                    temp = x[i*95:l- (5 - l%5)]
                    for c in range(0,(5 - l%5)):
                        temp = temp + '*'
                s = bytes(temp,'utf-8')
                buf.append(s)
        else:
            buf.append(byte_msg)
        return buf

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


# main path
async def qLearning():
    # qlearning definition
    env = Environment()
    env._seed(10)
    # learning parameters
    M = 6 #number of episodes, each episode made of three actions
    m = 1
    gamma = 0.9
    # initial Q function, it is a vector containg vectors of actions' values for each state
    Q = [[0.0,0.0,0.0], [0.0,0.0,0.0],[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]]

    robot = Connection('Spiky',"6E400001-B5A3-F393-E0A9-E50E24DCCA9E","6E400002-B5A3-F393-E0A9-E50E24DCCA9E", "6E400003-B5A3-F393-E0A9-E50E24DCCA9E")
    try:
        # connect to the robot
        await robot.connect()
        print('click the robot to start')
        await robot.getData()
        # waiting for the starting function
        print('robot needs to be initiated')
        await robot.getData()
        print('Starting...')
        # qlearning main loop
        while m < M:
            time.sleep(1)
            await robot.send_message(b'ack') # await to make the cycles on the two devices synchronized
            alpha = (1 - m/M)
            eps = (1 - m/M) ** 2
            # encode the Q matrix and send to the robot
            msg = encode_matrix(Q)
            # poichè ho riscontrato l'impossibilità di inviare messaggi più lunghi di 100 bytes, genero un 'buffer' contenente il messaggio separato
            # a blocchi di 100 bytes, qualora risulti di dimensione maggiore a 100 bytes
            buf = buffer(msg)
            await robot.getData()
            # ack to give the robot freedom to read
            await robot.send_message(b'acQ')
            print('Send Q...')
            l = len(buf) # calcolo la lunghezza del buffer (numero di messaggi in cui è stato diviso quello generale)
            a = str(l)
            l_b = bytes(a,'utf-8')
            await robot.send_message(l_b) # lo invio al robot
            for b in buf: # procedo con un invio sequenziale
                await robot.getData()
                await robot.send_message(b)
                await robot.getData()
            time.sleep(1)
            await robot.getData()
            time.sleep(1)
            msg_eps = str(eps)
            l = len(msg_eps)
            resto = 5 - l%5
            for x in range(0,resto):
                msg_eps = msg_eps + '*'
            # ack to give the robot freedom to read
            await robot.send_message(b'acE')
            print('Send eps...')
            await robot.send_message(bytes(msg_eps,'utf-8'))
            time.sleep(1)
            await robot.getData()
            print('Now wait the robot to finish its own computation')
            #while True:
            await robot.getData() # here it will come the finish ack from the robot
            time.sleep(4)
            print('Processing the episode...')
            results_msg = ''
            temp = await robot.getData()
            while True:
                results_msg = results_msg + temp
                if robot.queue.empty():
                    break
                else:
                    temp = await robot.getData()

            results = decode_matrix(results_msg)
            # 4. update the Q function
            for i in range(0,3):
                s = int(float(results[i][0]))
                a = int(float(results[i][1]))
                r = float(results[i][2])
                s_prime = s+a+1
                Q[s][a] = Q[s][a] + alpha*(r + gamma*(np.max(Q[s_prime]) - Q[s][a]))
            # ack to let the robot know that the computation is ended and a new cycle can begin
            print('Computation done')
            m += 1
            await robot.send_message(b'ac1')
        print('The final Q function is:')
        print(Q)

    except Exception as e:
        print(e)
    finally:
        await robot.disconnect()


async def main():
    def buffer(byte_msg):
        l = len(byte_msg)
        parts = int(l/95)
        buf = []
        x = str(byte_msg,'utf-8')
        if parts > 1:
            for i in range(0,parts):
                temp = x[i*95:(i+1)*95] + '*****'
                s = bytes(temp,'utf-8')
                buf.append(s)
        else:
            buf.append(byte_msg)
        return buf
    robot = Connection('Spiky',"6E400001-B5A3-F393-E0A9-E50E24DCCA9E","6E400002-B5A3-F393-E0A9-E50E24DCCA9E", "6E400003-B5A3-F393-E0A9-E50E24DCCA9E")
    try:
        await robot.connect()
        print('click the robot to start')
        await robot.getData()
        # waiting for the starting function
        print('robot needs to be initiated')
        await robot.getData()
        print('Starting...')
        input = ''
        # qlearning main loop
        while input != 'exit':
            data = 'abcd-abcd-abcd-abcd-abcd-*****'
            d = bytes(data,'utf-8')
            b = buffer(d)
            l = len(b)
            a = str(l)
            l_b = bytes(a,'utf-8')
            await robot.send_message(l_b)
            for bb in b:
                await robot.send_message(bb)
                await robot.getData()
    except Exception as e:
        await robot.disconnect()


asyncio.run(qLearning())