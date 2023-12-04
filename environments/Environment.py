from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, ColorSensor, UltrasonicSensor, ForceSensor
from pybricks.parameters import Button, Color, Direction, Port, Side, Stop
from pybricks.robotics import DriveBase
from pybricks.tools import wait, StopWatch
import urandom
import umath
from pybricks.geometry import Matrix

#hub = PrimeHub()

class Robot():

    def __init__(self,name):
        self.hub = PrimeHub()
        self.distance_sensor = UltrasonicSensor(Port.F)
        self.right_motor = Motor(Port.C)
        self.left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)
        self.moving_motors = DriveBase(self.left_motor, self.right_motor, wheel_diameter=56, axle_track=112)
        self.name = name


class Environment():

    def __init__(self,name):
        self.n_actions = 3
        self.n_states = 6
        self.allowed_actions = [[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]]
        self.mu = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.gamma = 0.9
        self.hub = Robot(name)

    def list_states(self):
        for i in range(0,self.n_states):
            print('State '+str(i)+': The robot is '+ str(40-i*10) + ' cm distant from the obstacle.' )

    def list_actions(self):
        for i in range(0,self.n_actions):
            print('Action '+str(i) + ': the robot moves foreward of '+str((i+1)*10)+' cm.')

    # method to set the environment random seed
    def _seed(self,seed):
        urandom.seed(seed)

    # method to reset the environment to the initial state
    def reset(self):
        s = 0
        return s

    # method to perform an environemnt transition
    def transition_model(self,s,a):
        eps = 0.1
        s_prime = s + a + 1
        s_prime = umath.floor(s_prime)
        if s == 2 and a == 0:
            inst_rew = 1.0
        else:
            inst_rew = -eps
        return s_prime, inst_rew

    def transition_model_no_robot(self,s,a):
        eps = 0.1
        s_prime = s + a + 1
        s_prime = umath.floor(s_prime)
        if s == 2 and a == 0:
            inst_rew = 1.0
        else:
            inst_rew = -eps
        return s_prime, inst_rew

    def action_execution(self,a):
        if a == 0:
            self.hub.moving_motors.straight(100)
        if a == 1:
            self.hub.moving_motors.straight(200)
        if a == 2:
            self.hub.moving_motors.straight(300)

    def comeback_function(self):
        d = self.hub.distance_sensor.distance()
        distance_to_do = 400 - d
        self.hub.moving_motors.straight(-distance_to_do)



def eps_greedy(Q, eps, allowed_actions, s):
    actions = []
    no_actions = []
    i = 0
    for a in allowed_actions:
        if a == 1:
            actions.append(i)
        else:
            no_actions.append(i)
        i += 1
    if urandom.random() <= eps:
        a = urandom.choice(actions)
    else:
        Q_s = Q[s]
        for na in no_actions:
            Q_s[na] = -1000
        a = Q_s.index(max(Q_s))
    return int(a)

def qLearning():
    env = Environment('Spiky')
    while env.hub.distance_sensor.distance() > 400:
        env.hub.moving_motors.drive(20,0)
        env.hub.moving_motors.stop()
    env._seed(10)
    # learning parameters
    M = 6 #number of episodes, each episode made of three actions
    m = 1
    # initial Q function, it is a vector containg vectors of actions' values for each state
    Q = [[0.0,0.0,0.0], [0.0,0.0,0.0],[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]]

    #Q-learning main loop
    while m<M:
        # initial state and action
        s = env.reset()
        a = eps_greedy(Q, 1.0, env.allowed_actions[s], s)
        alpha = (1 - m/M)
        eps = (1 - m/M) ** 2
        # execute an entire episode of three actions
        for i in range(0,3):
            print(i)
            s_prime, reward = env.transition_model_no_robot(s,a)
            # policy improvement step
            a_prime = eps_greedy(Q,eps,env.allowed_actions[s_prime], s_prime)
            #Q-learning update
            Q_s = Q[s]
            Q_sprime = Q[s_prime]
            Q_s[a] = Q_s[a] + alpha*(reward + env.gamma*max(Q_sprime) - Q_s[a])
            # make the robot execute the chosen action
            env.action_execution(a)
            #next iteration
            s = s_prime
            a = a_prime
        m += 1
        Q_matrix = Matrix(Q)
        print('Q matrix updated:')
        print(Q_matrix)
        # at the end of the episode the robot come back to the initial position/state
        env.comeback_function()
    Q_matrix = Matrix(Q)
    print('Final Q matrix:')
    print(Q_matrix)
    return Q

Q = qLearning()