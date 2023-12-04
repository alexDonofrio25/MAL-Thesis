from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, ColorSensor, UltrasonicSensor, ForceSensor
from pybricks.parameters import Button, Color, Direction, Port, Side, Stop, Icon
from pybricks.robotics import DriveBase
from pybricks.tools import wait, StopWatch
import urandom
import umath
from pybricks.geometry import Matrix

class Robot():

    def __init__(self,name):
        self.hub = PrimeHub()
        self.force_sensor = ForceSensor(Port.F)
        self.right_motor = Motor(Port.C)
        self.left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)
        self.moving_motors = DriveBase(self.left_motor, self.right_motor, wheel_diameter=88, axle_track=144.75)
        self.color_sensor = ColorSensor(Port.B) # da montare sul robot
        self.name = name
        self.angle = 0
        self.moving_motors.settings(500,400,360,180)

    def display_text(self,text):
        if isinstance(text,str) == False:
            text = str(text)
        self.hub.display.text(text)

    def display_arrows(self,action):
        if action == 0:
            if self.angle == 0:
                self.hub.display.icon(Icon.ARROW_LEFT)
            elif self.angle == 90:
                self.hub.display.icon(Icon.ARROW_DOWN)
            elif self.angle == -90:
                self.hub.display.icon(Icon.ARROW_UP)
        elif action == 1:
            if self.angle == 0:
                self.hub.display.icon(Icon.ARROW_RIGHT)
            elif self.angle == 90:
                self.hub.display.icon(Icon.ARROW_UP)
            elif self.angle == -90:
                self.hub.display.icon(Icon.ARROW_DOWN)
        elif action == 2:
            if self.angle == 0:
                self.hub.display.icon(Icon.ARROW_DOWN)
            elif self.angle == 90:
                self.hub.display.icon(Icon.ARROW_RIGHT)
            elif self.angle == -90:
                self.hub.display.icon(Icon.ARROW_LEFT)
        elif action == 3:
            if self.angle == 0:
                self.hub.display.icon(Icon.ARROW_UP)
            elif self.angle == 90:
                self.hub.display.icon(Icon.ARROW_LEFT)
            elif self.angle == -90:
                self.hub.display.icon(Icon.ARROW_RIGHT)

    def get_angle(self):
        return self.angle

    def set_angle(self,a):
        self.angle = a

    def turn_right(self,deg):
        #self.moving_motors.reset()
        teta = self.moving_motors.state()[2]
        while self.moving_motors.state()[2] < teta + deg:
            if self.moving_motors.state()[2] < teta + 0.8*deg :
                self.moving_motors.turn(20)
            else:
                self.moving_motors.turn(2)
        self.moving_motors.stop()

    def turn_left(self, deg):
        #self.moving_motors.reset()
        teta = self.moving_motors.state()[2]
        while self.moving_motors.state()[2] > teta - deg:
            if self.moving_motors.state()[2] > teta -0.8*deg :
                self.moving_motors.turn(-20)
            else:
                self.moving_motors.turn(-2)
        self.moving_motors.stop()

    # spiegazione dei movimenti: ogni volta che il robot esegue un'azione, attraverso il giroscopio controlla il suo angolo
    # e lo usa come sfasamento per la rotazione/movimento dritto.
    # teta è l'angolo di sfasamento
    # la rotazione di +- 90° è una rotazione controllata dove il robot gira di 20° per volta fino a raggiungere l'80% della rotazione
    # richiesta e poi completa il movimento girando di 2° per volta per aumentare la precisione sull'angolo finale
    def move_up(self,dis):
        teta = self.moving_motors.state()[2]
        if self.angle == 0:
            self.moving_motors.turn(-teta)
            wait(300)
            self.moving_motors.straight(-dis,Stop.BRAKE)
        elif self.angle == 90:
            self.turn_left(teta)
            wait(300)
            self.moving_motors.straight(-dis,Stop.BRAKE)
        elif self.angle == -90:
            self.turn_right(-teta)
            wait(300)
            self.moving_motors.straight(-dis,Stop.BRAKE)
        self.angle = 0

    def move_down(self,dis):
        teta = self.moving_motors.state()[2]
        if self.angle == 0:
            self.moving_motors.turn(-teta)
            wait(300)
            self.moving_motors.straight(dis,Stop.BRAKE)
        elif self.angle == 90:
            self.turn_left(teta)
            wait(300)
            self.moving_motors.straight(dis,Stop.BRAKE)
        elif self.angle == -90:
            self.turn_right(-teta)
            wait(300)
            self.moving_motors.straight(dis,Stop.BRAKE)
        self.angle = 0

    def move_right(self, dis):
        teta = self.moving_motors.state()[2]
        if self.angle == 0:
            self.turn_right(90 - teta)
            wait(300)
            self.moving_motors.straight(dis,Stop.BRAKE)
            self.angle = 90
        elif self.angle == 90:
            self.moving_motors.turn(90-teta)
            wait(300)
            self.moving_motors.straight(dis,Stop.BRAKE)
        elif self.angle == -90:
            self.moving_motors.turn(-90-teta)
            wait(300)
            self.moving_motors.straight(-dis,Stop.BRAKE)

    def move_left(self, dis):
        teta = self.moving_motors.state()[2]
        if self.angle == 0:
            self.turn_left(90 - teta)
            wait(300)
            self.moving_motors.straight(dis,Stop.BRAKE)
            self.angle = -90
        elif self.angle == 90:
            self.moving_motors.turn(90-teta)
            wait(300)
            self.moving_motors.straight(-dis,Stop.BRAKE)
        elif self.angle == -90:
            self.moving_motors.turn(-90-teta)
            wait(300)
            self.moving_motors.straight(dis,Stop.BRAKE)

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
        #self.grid = [[2,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,3]]
        self.hub = Robot('Spiky')

    def set_start(self,start):
        # with this method you can set the starting point of the grid
        # as a tuple of row and column
        self.start = start
        self.grid[start[0]][start[1]] = 2

    def set_goal(self,goal):
        # with this method you can set the starting point of the grid
        # as a tuple of row and column
        self.goal = goal
        self.grid[goal[0],goal[1]] = 3

    def set_position(self,pos):
        self.position = pos

    def get_position(self):
        return self.position

    def update_position(self,action):
        if action == 0:
            p = self.get_position()
            new_p = [p[0]-1,p[1]]
            self.set_position(new_p)
        elif action == 1:
            p = self.get_position()
            new_p = [p[0]+1,p[1]]
            self.set_position(new_p)
        elif action == 2:
            p = self.get_position()
            new_p = [p[0],p[1]-1]
            self.set_position(new_p)
        elif action == 3:
            p = self.get_position()
            new_p = [p[0],p[1]+1]
            self.set_position(new_p)

    def transition_model(self,a):
        # action encoding:
        # 0: up
        # 1: down
        # 2: right
        # 3: left
        s_tuple = self.get_position()
        s = self.tuple_to_state(s_tuple)
        self.update_position(a)
        s_prime_tuple = self.get_position()
        s_prime = self.tuple_to_state(s_prime_tuple)
        if s_prime_tuple[0] < 0 or s_prime_tuple[1] < 0 or s_prime_tuple[0] >= self.nRows or s_prime_tuple[1] >= self.nCols:
            s_prime = s
            self.set_position(s_tuple)
        color = self.hub.color_sensor.color(True)
        if color == Color.BLUE:
            reward = -0.1
        elif color == Color.RED:  # set the second color according to the borders
            reward = -1
            s_prime = s
            self.set_position(s_tuple)
        elif color == Color.GREEN:
            reward = 1
        elif color == Color.NONE:
            if s_prime_tuple[0] < 0 or s_prime_tuple[1] < 0 or s_prime_tuple[0] >= self.nRows or s_prime_tuple[1] >= self.nCols:
                reward = -1
                s_prime = s
                self.set_position(s_tuple)
            else:
                reward = -0.1
        return s_prime,reward

    def transition_model_new(self,a):
        # action encoding:
        # 0: up
        # 1: down
        # 2: right
        # 3: left
        s_tuple = self.get_position()
        s = self.tuple_to_state(s_tuple)
        self.update_position(a)
        s_prime_tuple = self.get_position()
        s_prime = self.tuple_to_state(s_prime_tuple)
        if s_prime_tuple[0] < 0 or s_prime_tuple[1] < 0 or s_prime_tuple[0] >= self.nRows or s_prime_tuple[1] >= self.nCols:
            s_prime = s
            self.set_position(s_tuple)
            reward = -1
            self.hub.hub.display.icon(Icon.FALSE)
        else:
            self.do_move(a,240,240)
            wait(500)
            color = self.hub.color_sensor.color(True)
            if color == Color.BLUE or color == Color.NONE:
                reward = -0.1
            elif color == Color.RED:  # set the second color according to the borders
                reward = -1
                s_prime = s
                self.undo_move(a,240,240)
                self.set_position(s_tuple)
            elif color == Color.GREEN:
                reward = 1
        return s_prime,reward

    def tuple_to_state(self,t):
        s = t[0]*self.nCols + t[1]
        return s

    def do_move(self,a,d1,d2):
        if a == 0:
            self.hub.move_up(d1)
        elif a == 1:
            self.hub.move_down(d1)
        elif a == 2:
            self.hub.move_right(d2)
        elif a == 3:
            self.hub.move_left(d2)

    def undo_move(self,a,d1,d2):
        if a == 0:
            self.hub.move_down(d1)
        elif a ==1:
            self.hub.move_up(d1)
        elif a == 2:
            self.hub.move_left(d2)
        elif a == 3:
            self.hub.move_right(d2)

    def comeback_function(self,d1,d2):
        #self.hub.moving_motors.settings(200,100,720,360)
        p = self.get_position()
        print(p)
        up = p[0]*d1
        right = p[1]*d2
        for v in range(0,p[0]):
            self.hub.move_up(d1)
        if right != 0:
            for b in range(0,p[1]):
                self.hub.move_right(d2)
            teta = self.hub.moving_motors.state()[2]
            #self.hub.moving_motors.straight(-15)
            wait(500)
            self.hub.turn_left(90-(teta-90))
            wait(500)
            #self.hub.moving_motors.straight(-30)
        self.hub.angle = 0
    # method to set the environment random seed
    def _seed(self, seed):
        urandom.seed(seed)

def eps_greedy(s, Q, eps):
    actions = [0,1,2,3]
    if urandom.random() < eps:
        a = urandom.choice(actions)
    else:
        Q_s = Q[s]
        a = Q_s.index(max(Q_s))
    return int(a)

def qLearning():
    env = Environment()
    env._seed(10)
    # learning parameters
    M = 80
    m = 1
    k = 9 # length of the episode
    # initial Q function
    Q = []
    for i in range(0,env.nS):
        Q.append([0,0,0,0])
    #first action
    watch = StopWatch()
    print(watch.time())
    while m <= M:
        print('Iteration number: ',m)
        env.set_position(env.start)
        s = env.tuple_to_state(env.start)
        a = eps_greedy(s, Q, 1)
        alpha = (1 - m/M)
        eps = (1 - m/M) ** 4
        for i in range (0,k):
            env.hub.hub.light.on(Color.GREEN)
            env.hub.display_text(s)
            wait(300)
            env.hub.display_arrows(a)
            wait(300)
            #env.do_move(a,240) # add distance
            s_prime, reward = env.transition_model_new(a)
            #if s_prime == s:
                #env.undo_move(a,240)
            # policy improvement step
            a_prime = eps_greedy(s_prime,Q,eps)
            # Q-learning update
            Q_s = Q[s]
            Q_sprime = Q[s_prime]
            Q_s[a] = Q_s[a] + alpha * (reward + env.gamma * max(Q_sprime) - Q_s[a])
            # update the environment position
            s = s_prime
            a = a_prime
            if s == 24:
                env.hub.hub.speaker.beep(200,60)
                break
        env.hub.hub.light.on(Color.MAGENTA)
        env.comeback_function(240,240)
        # next iteration
        m = m + 1
        print('Q matrix updated:')
        print(Matrix(Q))
        print('----------------------------------------------')
    print('Final Q function:\n',Matrix(Q))
    print(watch.time())
    return Q

Q = qLearning()