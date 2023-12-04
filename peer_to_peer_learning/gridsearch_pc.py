import asyncio
from bleak import BleakScanner, BleakClient
import time
import numpy as np
from eseguibile_pc import decode_matrix,encode_matrix,buffer, Connection

class GridSearch():
    def __init__(self):
        self.nRows = 5
        self.nCols = 5
        self.nS = self.nRows*self.nCols
        self.nA = 4
        self.nO = 4 # number of ostacle on the grid
        self.actual_states = (self.nRows*self.nCols) - self.nO
        mu = []
        for i in range(0,self.actual_states):
            if i == 0:
                mu.append(1.0)
            else:
                mu.append(0.0)
        self.mu = mu
        self.gamma = 0.9

    # method to set the environment random seed
    def _seed(self, seed):
        np.random.seed(seed)

async def gridsearch_qlearning():
    env = GridSearch()
    env._seed(15)
    # learning parameters
    M = 150
    m = 1
    k = 8 # length of the episode
    # initial Q function
    Q = np.zeros((env.nS,4))

    robot = Connection('Spiky',"6E400001-B5A3-F393-E0A9-E50E24DCCA9E","6E400002-B5A3-F393-E0A9-E50E24DCCA9E", "6E400003-B5A3-F393-E0A9-E50E24DCCA9E")
    try:
        # connect to the robot
        await robot.connect()
        print('click the robot to start')
        await robot.getData()
        # qlearning main loop
        while m < M:
            # waiting for the starting function
            print('robot needs to be initiated')
            await robot.getData()
            print('Starting...')
            time.sleep(1)
            await robot.send_message(b'ack') # await to make the cycles on the two devices synchronized
            alpha = (1 - m/M)
            eps = (1 - m/M) ** 2
            # encode the Q matrix and send to the robot
            q_normal = []
            for i in range(0,env.nS):
                q_row = []
                for j in range(0,4):
                    q_row.append(Q[i,j])
                q_normal.append(q_row)
            msg = encode_matrix(q_normal)
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
            for i in range(0,8):
                s = int(float(results[i][0]))
                a = int(float(results[i][1]))
                r = float(results[i][2])
                s_prime = s+a+1
                Q[s,a] = Q[s,a] + alpha*(r + env.gamma*(np.max(Q[s_prime :]) - Q[s,a]))
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

def main():
    Q = np.zeros((25,4))
    q_normal = []
    for i in range(0,25):
        q_row = []
        for j in range(0,4):
            q_row.append(Q[i,j])
        q_normal.append(q_row)
    msg = encode_matrix(q_normal)

#main()
asyncio.run(gridsearch_qlearning())