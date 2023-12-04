# computer

import asyncio
from bleak import BleakScanner, BleakClient
import time
import numpy as np

# communication step
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# Replace this with the name of your hub if you changed
# it when installing the Pybricks firmware.
HUB_NAME = "Spiky"


def hub_filter(device, ad):
    return device.name and device.name.lower() == HUB_NAME.lower()


def handle_disconnect(_):
    print("Hub was disconnected.")


def handle_rx(_, data: bytearray):
    print("Received:", data)

# environment

class Environment():
    def __init__(self):
        self.n_actions = 3
        self.n_states = 6
        self.allowed_actions = [[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]]
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
            v_str = v_str + els + '-'
        nrs = str(len(Q))
        ncs = str(len(Q[0]))
        msg = nrs + '|' + ncs + '|' + v_str + '*****'
        # write the encoding function from computer to bytearray
        # Q has to be in matrix format
        byte_msg = bytes(msg,'utf-8')
        return byte_msg

def decode_matrix(msg):
    msg_split = msg.rsplit('|')
    nr = int(msg_split[0])
    nc = int(msg_split[1])
    v_string = msg_split[2]
    v = v_string.rsplit('-')
    Q = []
    for i in range(0,nr):
        q = []
        for j in range(0,nc):
            q.append(v[j+i*nc])
        Q.append(q)
    return Q

async def main():
    # Find the device and initialize client.
    device = await BleakScanner.find_device_by_filter(hub_filter)
    client = BleakClient(device, disconnected_callback=handle_disconnect)

    # Shorthand for sending some data to the hub.
    async def send(client, data):
        await client.write_gatt_char(rx_char, data)

    def read():
        dec = client.read_gatt_char(UART_TX_CHAR_UUID)
        data = dec.decode()
        return data

    def idle():
        ack = None
        try:
            ack = read()
            return False
        except Exception as e:
            return True

    try:

        # qlearning definition
        env = Environment()
        env._seed(10)
        # learning parameters
        M = 6 #number of episodes, each episode made of three actions
        m = 1
        gamma = 0.9
        # initial Q function, it is a vector containg vectors of actions' values for each state
        Q = [[0.0,0.0,0.0], [0.0,0.0,0.0],[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]]

        # Connect and get services.
        await client.connect()
        await client.start_notify(UART_TX_CHAR_UUID, handle_rx)
        nus = client.services.get_service(UART_SERVICE_UUID)
        rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)

        flag = True
        while flag:
            print("Start the program on the hub now with the button.")
            flag = idle()
            time.sleep(2)

        # 2. implement the starting function, an action that once happened sends a message to the computer telling
        # the execution is started and it can start sending message
        flag = True
        while flag:
            print("Starting...")
            flag = idle()
            time.sleep(2)

        # qLearning main loop
        while m<M:
            alpha = (1 - m/M)
            eps = (1 - m/M) ** 2
            # in this version, the only computation that the computer has to do is the Q Matrix updating
            # 1. send the Q matrix to the hub:
            #await send(client,b'ack')
            #time.sleep(1)
            msg = encode_matrix(Q)
            send(client,msg)
            time.sleep(1)
            #await send(client,b'ack')
            time.sleep(1)
            msg_eps = str(eps)
            msg_eps = msg_eps + '*****'
            send(client,bytes(msg_eps,'utf-8'))
            time.sleep(1)
            # 2.wait until the routine is not ended
            flag = True
            while flag:
                print("Cooking...")
                flag = idle()
                asyncio.sleep(5)
                time.sleep(5)
            # 3. read the results of the robot job, they consist in state,action tuples and the linked reward
            results_msg = read()
            results = decode_matrix(results_msg)
            # 4. update the Q function
            for i in range(0,3):
                s = results[i][0]
                a = results[i][1]
                r = results[i][2]
                s_prime = s+a+1
                Q[s][a] = Q[s][a] + alpha*(r + gamma*(np.max(Q[s_prime]) - Q[s][a]))
            send(client,b'ack')
        print('The final Q function is:')
        print(Q)


    except Exception as e:
        # Handle exceptions.
        print(e)
    finally:
        # Disconnect when we are done.
        await client.disconnect()


# Run the main async program.
asyncio.run(main())