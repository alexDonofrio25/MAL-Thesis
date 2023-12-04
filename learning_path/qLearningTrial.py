import numpy as np
import time
import asyncio
from bleak import BleakScanner, BleakClient
import bleak.backends.service as bs

# connection variables
# Nordic UART Service setup
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
HUB_NAME = "Spiky" # the name of the hub

# functions
def handle_disconnect(_):
    print("Hub was disconnected.")


def handle_rx(_, data: bytearray):
    print("Received:", data)

async def bleak_client():
    device = await BleakScanner.find_device_by_name(HUB_NAME)
    client = BleakClient(device, disconnected_callback=handle_disconnect)
    print(client)
    return client



async def bleak_connection(client):
    print('client: ' + str(client))
    async with client:
        await client.connect()
        check =client.is_connected
        print('Connected:' + str(check))
        print('Client information:')
        print('Address:' + str(client.address))
        print('MTU size:' + str(client.mtu_size))
        return client
# I'm defining a small problem of Reinforcement Learning where a robot starting from a pre-defined position
# should stop its movements 50 cm before an obstacle in only three moves

class Environment(object):

  def __init__(self,distance):
    #number of actions and states
    self.nA = 3
    self.nS = 7
    self.allowed_actions = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1]])
    #distance from the target
    #self.dis = distance
    #self.current_dis = distance
    # initial state distribution and discount factor
    self.mu = np.array([1, 0., 0., 0., 0., 0., 0.])
    self.gamma = 0.9
    # transition model (SA rows, S columns)
    self.P = np.array([[0.,1,0.,0.,0.,0.,0.],
                       [1,0.,0.,0.,0.,0.,0.],
                       [1,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,1,0.,0.,0.,0.],
                       [0.,1,0.,0.,0.,0.,0.],
                       [0.,0.,0.,1,0.,0.,0.],
                       [0.,1,0.,0.,0.,0.,0.],
                       [0.,0.,1,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,1,0.,0.],
                       [0.,0.,0.,1,0.,0.,0.],
                       [0.,0.,0.,0.,0.,1,0.],
                       [0.,0.,0.,1,0.,0.,0.],
                       [0.,0.,0.,0.,1,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,1],
                       [0.,0.,0.,0.,0.,1,0.],
                       [1,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,1,0.],
                       [0.,0.,0.,0.,0.,0.,1]
                       ])
    # immediate reward (SA rows, S columns)
    self.R = np.array([[0.,1,0.,0.,0.,0.,0.],
                       [-2,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,1,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,1,0.,0.,0.],
                       [0.,-2,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,1,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,1,0.],
                       [0.,0.,0.,-2,0.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,0.,1],
                       [0.,0.,0.,0.,0.,0.,0.],
                       [1,0.,0.,0.,0.,0.,0.],
                       [0.,0.,0.,0.,0.,-2,0.],
                       [0.,0.,0.,0.,0.,0.,0.]
                       ])

# method to set the environment random seed
  def _seed(self, seed):
    np.random.seed(seed)
# method to reset the environment to the initial state
  def reset(self):
    self.s = s = np.random.choice(self.nS, p=self.mu)
    return s
# method to perform an environemnt transition
  def transition_model(self, a):
    sa = self.s * self.nA + a
    self.s = s_prime = np.random.choice(self.nS, p=self.P[sa, :])
    inst_rew = self.R[sa, s_prime]
    return s_prime, inst_rew

# definition of the greedy policy for our model
def eps_greedy(s, Q, eps, allowed_actions):
  if np.random.rand() <= eps:
    actions = np.where(allowed_actions)
    actions = actions[0] # just keep the indices of the allowed actions
    a = np.random.choice(actions, p=(np.ones(len(actions)) / len(actions)))
  else:
    Q_s = Q[s, :].copy()
    Q_s[allowed_actions == 0] = - np.inf
    a = np.argmax(Q_s)
  return a

def qLearning():
    #implementing the Q-Learning algorithm
        policy = []
        policy_list = []
        env1 = Environment(100)
        env1._seed(10)
        # learning parameters
        M = 1000
        m = 1
        # initial Q function
        Q = np.zeros((env1.nS, env1.nA))
        # initial state and action
        s = env1.reset()
        a = eps_greedy(s, Q, 1., env1.allowed_actions[s])
        # Q-learning main loop
        while m < M:
            alpha = (1 - m/M)
            eps = (1 - m/M) ** 2
            # environment step
            s_prime, r = env1.transition_model(a)
            #print('s_prime:' + str(s_prime))
            # policy improvement step
            a_prime = eps_greedy(s, Q, eps, env1.allowed_actions[s_prime])
            # Q-learning update
            Q[s, a] = Q[s, a] + alpha * (r + env1.gamma * np.max(Q[s_prime, :]) - Q[s, a])
            # save the policy
            if s != 0:
                policy.append(a)
            elif s == 0 and a == 1:
                policy.append(a)
            elif s == 0 and a != 1:
                if policy != []:
                    policy_list.append(policy)
                    policy = [a]
            # next iteration
            m = m + 1
            s = s_prime
            a = a_prime
        return Q,policy_list

def main_without_connection():
    Q, policy_list = qLearning()
    print(policy_list[0])
    print(policy_list[len(policy_list) - 1])
    print('The final Q function is:\n', Q)
    bp = best_policy(Q)
    print('The best policy is:')
    print(bp)

def best_policy(Q):
    best_policy=[]
    for q in Q:
        max = q[0]
        i = 0
        j = 1
        while (j < len(q)):
            if q[j] > max:
                max = q[j]
                i = j
            j += 1
        best_policy.append(i)
    return best_policy

async def main() :
    try:
        #generate the client
        client = await bleak_client()
        #creating the connection
        await client.connect()
        await client.start_notify(UART_TX_CHAR_UUID, handle_rx)
        nus = client.services.get_service(UART_SERVICE_UUID)
        rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)

        # Shorthand for sending some data to the hub.
        async def send(client, data):
            await client.write_gatt_char(rx_char, data)

        async def action_encoder(client, action):
            if action == 0:
                await send(client,b'light')
                time.sleep(2)
                await send(client,b'fwd20')
                time.sleep(2)
            elif action == 1:
                await send(client,b'light')
                time.sleep(2)
                await send(client,b'turnR')
                time.sleep(2)
            elif action == 2:
                await send(client,b'light')
                time.sleep(1)

        # Tell user to start program on the hub.
        print("Start the program on the hub now with the button.")
        time.sleep(5)
        print('Starting...')

        Q, policy_list = qLearning()
        print(policy_list)

        l = len(policy_list)
        for x in policy_list[0]:
            await action_encoder(client,x)
        time.sleep(2)
        await send(client,b'light')
        await send(client,b'light')
        time.sleep(2)
        for x in policy_list[l-1]:
            await action_encoder(client,x)
    except Exception as e:
        # Handle exceptions.
        print(e)
    finally:
        # Disconnect when we are done.
        await client.disconnect()
        print('The final Q function is:\n', Q)

main_without_connection()
#asyncio.run(main())