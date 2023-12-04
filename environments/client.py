# in this case the server is intended as the computer
#from HubConnection import HubConnection
from bleak import BleakScanner, BleakClient
import asyncio
import time
import numpy as np

class Client():

    def __init__(self,hub_name):
        self.hub = hub_name
        self.client = None

    def hub_filter(self,device, ad):
            return device.name and device.name.lower() == self.hub.lower()


    def handle_disconnect(self, _):
        print("Hub was disconnected.")


    async def handle_rx(self, _, data: bytearray):
        print("Received:", data)

    # Shorthand for sending some data to the hub.
    async def send(self, rx_char, data):
        await self.client.write_gatt_char(rx_char, data)

    async def read(self):
        UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        dec = await self.client.read_gatt_char(UART_TX_CHAR_UUID)
        data = dec.decode()
        return data

    async def connect(self):

        UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
        UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

        try:
            # Find the device and initialize client.
            device = await BleakScanner.find_device_by_filter(self.hub_filter)
            self.client = BleakClient(device, disconnected_callback=self.handle_disconnect)

            # Connect and get services.
            await self.client.connect()
            await self.client.start_notify(UART_TX_CHAR_UUID, self.handle_rx)
            nus = self.client.services.get_service(UART_SERVICE_UUID)
            rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)
        except Exception as e:
            # Handle exceptions.
            print(e)
        finally:
            return rx_char

    async def disconnect(self):
        await self.client.disconnect()

    async def initiation_loop(self):
        # Tell user to start program on the hub.
        print("Start the program on the hub now with the button.")
        ack = None
        time.sleep(5)
        while ack == None:
            try:
                print('Starting...')
                ack = await self.read()
                if ack != None:
                    print('ack received')
                else:
                    print('no ack')
            except Exception as e:
                print(e)



    async def wait_until(self):
        str = None
        print('Make Spiky look something close to it...')
        while str == None:
            try:
                str = await self.read()
            except Exception as e:
                print ('Nothing happened..')
            finally:
                time.sleep(2)
        print ('----------------------------------')
        print ('Communication start...')

    def encode_matrix(self, Q):
        # write the encoding function from computer to bytearray
        # Q has to be in matrix format
        byte_msg = bytearray('')
        return byte_msg

    def decode_matrix(self, msg):
        # write the decoding function from bytearray to computer
        #return Q_vector, Q
        return ''

    async def wait_Q(self, rx_char):
        Q_msg = None
        print('Computer is waiting the Q matrix')
        while Q_msg == None:
            try:
                Q_msg = await self.read()
            except Exception as e:
                print('No message arrived..')
            finally:
                time.sleep(2)
        await self.send(rx_char, b'ack')
        Q = self.decode_matrix(Q_msg)
        return Q

    async def send_Q(self, Q, rx_char):
        ack = None
        while ack == None:
            print('wait until the robot is ready')
            try:
                ack = await self.read()
            except Exception as e:
                print('No ack received')
            time.sleep(2)
        Q_msg = self.encode_matrix(Q)
        await self.send(rx_char, Q_msg)
        ack = None
        while ack == None:
            print('wait until the robot has received the data')
            try:
                ack = await self.read()
            except Exception as e:
                print('No ack received')
            time.sleep(2)



class Client_Environment():

    def __init__(self):
        # states and actions
        self.client = Client('Spiky')
        self.nS = 6
        self.nA = 3
        self.allowed_actions = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]])
        # initial state distribution and discount factor
        self.mu = np.array([1., 0., 0.])
        self.gamma = 0.9
        # transition model (SA rows, S columns)
            # the transittion model is handled in the robot itself
        # immediate reward (SA rows, S columns)
            # the immediate reward is handled in the robot itself

    # method to set the environment random seed
    def _seed(self, seed):
        np.random.seed(seed)

  # method to reset the environment to the initial state
    def reset(self):
        self.s = s = np.random.choice(self.nS, p=self.mu)
        return s

  # method to perform an environemnt transition, the environment transition is made in the robot itself
    def transition_model(self, a):
        sa = self.s * self.nA + a
        self.s = s_prime = np.random.choice(self.nS, p=self.P[sa, :])
        inst_rew = self.R[sa, s_prime]
        return s_prime, inst_rew


async def qLearning(rx_char):
    # instantiate the environment
    env = Client_Environment()
    env._seed(10)
    # learning parameters
    M = 5 # number of episodes to execute
    m = 1
    # initial Q function
    Q = np.zeros((env.nS, env.nA))
    while m < M:
        alpha = (1 - m/M)
        eps = (1 - m/M) ** 2
        # environment step: send the Q matrix to the robot and wait
        await env.client.send_Q(Q,rx_char)
        time.sleep(5)
        Q_vector = await env.client.wait_Q(rx_char)
        # now we compute the Q update
        Q_robot = np.array(Q_vector)
        s = 0
        for q in Q_robot:
            a = q[0]
            r = q[1]
            s_prime = s + a + 1
            Q[s,a] = Q[s, a] + alpha * (r + env.gamma * np.max(Q[s_prime, :]) - Q[s, a])
            s = s_prime
        m = m+1
    print('the final Q function is:\n', Q)
