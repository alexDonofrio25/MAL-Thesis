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
        self.angle = 0
        self.moving_motors.settings(200,120,50,90)

    def get_angle(self):
        return self.angle

    def set_angle(self,a):
        self.angle = a

    def move_up_advanced(self,dis):
        if self.angle == 0:
            self.moving_motors.straight(-dis)
            self.moving_motors.stop()
            wait(1000)
        elif self.angle == 90:
            self.moving_motors.turn(-77)
            self.moving_motors.stop()
            wait(1000)
            self.moving_motors.straight(-dis)
            self.moving_motors.stop()
            wait(1000)
            self.angle = 0
        elif self.angle == -90:
            self.moving_motors.turn(77)
            self.moving_motors.stop()
            wait(1000)
            self.moving_motors.straight(-dis)
            self.moving_motors.stop()
            wait(1000)
            self.angle = 0

    def move_up(self,dis):
        self.moving_motors.straight(-dis)
        wait(1000)

    def move_down_advanced(self,dis):
        if self.angle == 0:
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(1000)
        elif self.angle == 90:
            self.moving_motors.turn(-77)
            self.moving_motors.stop()
            wait(1000)
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(1000)
            self.angle = 0
        elif self.angle == -90:
            self.moving_motors.turn(77)
            self.moving_motors.stop()
            wait(1000)
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(1000)
            self.angle = 0

    def move_down(self,dis):
        self.moving_motors.straight(dis)
        wait(1000)

    def move_right_advanced(self,dis):
        if self.angle == 0:
            self.moving_motors.turn(77)
            self.moving_motors.stop()
            wait(1000)
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(1000)
            self.angle += 90
        elif self.angle == 90:
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(1000)
        elif self.angle == -90:
            self.moving_motors.turn(77)
            self.moving_motors.turn(77)
            self.moving_motors.stop()
            wait(1000)
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(1000)
            self.angle += 180

    def move_right(self,dis):
        self.moving_motors.turn(77)
        self.moving_motors.stop()
        wait(1000)
        self.moving_motors.straight(dis)
        self.moving_motors.stop()
        wait(1000)
        self.moving_motors.turn(-77)
        self.moving_motors.stop()
        wait(1000)

    def move_left_advanced(self,dis):
        if self.angle == 0:
            self.moving_motors.turn(-77)
            self.moving_motors.stop()
            wait(1000)
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(1000)
            self.angle += -90
        elif self.angle == -90:
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(1000)
        elif self.angle == 90:
            self.moving_motors.turn(-77)
            self.moving_motors.turn(-77)
            self.moving_motors.stop()
            wait(1000)
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(1000)
            self.angle += -180

    def move_left(self,dis):
        self.moving_motors.turn(-77)
        self.moving_motors.stop()
        wait(1000)
        self.moving_motors.straight(dis)
        self.moving_motors.stop()
        wait(1000)
        self.moving_motors.turn(77)
        self.moving_motors.stop()
        wait(1000)


class Environment():

    def __init__(self):
        self.nRows = 5
        self.nCols = 5
        self.nS = self.nRows*self.nCols
        self.nA = 4
        self.nO = 4 # number of ostacle on the grid
        self.start = [0,0]
        self.goal = [4,4]
        self.position = self.start
        # allowed actions is not necessary, the robot can take every action in every state
        self.actual_states = self.nS - self.nO
        mu = []
        for i in range(0,self.actual_states):
            if i == 0:
                mu.append(1.0)
            else:
                mu.append(0.0)
        self.mu = mu
        self.gamma = 0.9
        self.grid = [[2,0,0,0,0],[0,1,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[0,1,0,0,3]]
        self.hub = Robot('Spiky')


    def set_start(self,start):
        # with this method you can set the starting point of the grid
        # as a tuple of row and column
        self.start = start
        self.grid[start[0],start[1]] = 2

    def set_goal(self,goal):
        # with this method you can set the starting point of the grid
        # as a tuple of row and column
        self.goal = goal
        self.grid[goal[0],goal[1]] = 3

    def set_position(self,pos):
        self.position = pos
    #def create_grid(self):
    def get_position(self):
        return self.position

    def transition_model(self,a):
        # action encoding:
        # 0: up
        # 1: down
        # 2: left
        # 3: right
        eps = 0.1
        g = Matrix(self.grid)
        s = self.get_position()
        if a == 0:
            if s[0] - 1 >= 0 and g[s[0]-1,s[1]] != 1:
                s_prime = [s[0] - 1,s[1]]
                if g[s_prime[0],s_prime[1]] == 3:
                    inst_rew = 1
                else:
                    inst_rew = -eps
            else:
                s_prime = s
                inst_rew = -1
        elif a == 1:
            if s[0] + 1 < self.nRows and g[s[0]+1,s[1]] != 1:
                s_prime = [s[0] + 1,s[1]]
                if g[s_prime[0],s_prime[1]] == 3:
                    inst_rew = 1
                else:
                    inst_rew = -eps
            else:
                s_prime = s
                inst_rew = -1
        elif a == 2:
            if s[1] - 1 >= 0 and g[s[0],s[1] - 1] != 1:
                s_prime = [s[0],s[1]-1]
                if g[s_prime[0],s_prime[1]] == 3:
                    inst_rew = 1
                else:
                    inst_rew = -eps
            else:
                s_prime = s
                inst_rew = -1
        elif a == 3:
            if s[1] + 1 < self.nCols and g[s[0],s[1]+1] != 1:
                s_prime = [s[0],s[1]+1]
                if g[s_prime[0],s_prime[1]] == 3:
                    inst_rew = 1
                else:
                    inst_rew = -eps
            else:
                s_prime = s
                inst_rew = -1
        return s_prime,inst_rew

    def tuple_to_state(self,t):
        s = t[0]*self.nCols + t[1]
        return s

    def do_move(self,a,d):
        if a == 0:
            self.hub.move_up_advanced(d)
        elif a == 1:
            self.hub.move_down_advanced(d)
        elif a == 2:
            self.hub.move_right_advanced(d)
        elif a == 3:
            self.hub.move_left_advanced(d)

    def undo_move(self,a,d):
        if a == 0:
            self.hub.move_down_advanced(d)
        elif a ==1:
            self.hub.move_up_advanced(d)
        elif a == 2:
            self.hub.move_left_advanced(d)
        elif a == 3:
            self.hub.move_right_advanced(d)



    def comeback_function(self,d):
        p = self.get_position()
        print(p)
        up = p[0]*d
        right = p[1]*d
        self.hub.move_up_advanced(up)
        self.hub.move_right_advanced(right)
        self.hub.moving_motors.turn(-77)
        self.hub.angle = 0

    # method to set the environment random seed
    def _seed(self, seed):
        urandom.seed(seed)

def eps_greedy(s, Q, eps):
    actions = [0,1,2,3]
    if urandom.random() <= eps:
        a = urandom.choice(actions)
    else:
        Q_s = Q[s]
        a = Q_s.index(max(Q_s))
    return int(a)

def qLearning():
    env = Environment()
    env._seed(10)
    # learning parameters
    M = 150
    m = 1
    k = 8 # length of the episode
    # initial Q function
    Q = []
    for i in range(0,env.nS):
        Q.append([0,0,0,0])
    #first action

    while m <= M:
        env.set_position(env.start)
        s = env.tuple_to_state(env.start)
        a = eps_greedy(s, Q, 1)
        alpha = (1 - m/M)
        eps = (1 - m/M) ** 2
        for i in range (0,k):
            if m%30 == 0 or m == 1 or m == 4:
                print(m)
                print('step:',i)
                print('s:',s)
                print('a:',a)
            if m%30 == 0 or m == 1 or m == 4:
                env.do_move(a,153)
            s_prime_tuple, reward = env.transition_model(a)
            s_prime = env.tuple_to_state(s_prime_tuple)
            if m%30 == 0 or m == 1 or m == 4:
                if s == s_prime:
                    env.undo_move(a,153)
            # policy improvement step
            a_prime = eps_greedy(s_prime,Q,eps)
            # Q-learning update
            Q_s = Q[s]
            Q_sprime = Q[s_prime]
            Q_s[a] = Q_s[a] + alpha * (reward + env.gamma * max(Q_sprime) - Q_s[a])
            # update the environment position
            env.set_position(s_prime_tuple)
            s = s_prime
            a = a_prime
        if m%30 == 0 or m == 1 or m == 4:
            env.comeback_function(153)
        # next iteration
        m = m + 1
        #print('Q matrix updated:')
        #print(Q)
    print('Final Q function:\n',Q)
    return Q

Q = qLearning()