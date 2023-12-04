from pybricks.pupdevices import Motor, ForceSensor, UltrasonicSensor, ColorSensor
from pybricks.hubs import PrimeHub
from pybricks.parameters import Port, Icon, Direction, Color
from pybricks.tools import wait
from pybricks.geometry import Matrix
from pybricks.robotics import DriveBase
import urandom, umath
# Standard MicroPython modules
from usys import stdin, stdout
from uselect import poll

# utility functions
def decripting_function():
    msg = ''
    flag = True
    while flag:
        #x = str(i)
        #hub.display.text(x)
        cmd = stdin.buffer.read(5)
        t = repr(cmd)
        #hub.display.text(t)
        v = t.replace('b','')
        temp = v.replace("'",'')
        temp1 = ''
        for t in temp:
            if t == '*': # special char that indicates the end of a message
                flag = False
                break
            else:
                temp1 = temp1 + t
        msg = msg + temp1
    return msg

def decode_matrix(msg):
    if isinstance(msg,bytes):  # check if the input message is a bytearray, if yes, it casts it into a
        msg = str(msg,'utf-8')

    split = msg.rsplit('|')
    rows = int(float(split[0]))
    columns = int(float(split[1]))

    vector = list()

    v_split = split[2].rsplit('/')
    for v in v_split:
        if v != '':
            vector.append(float(v))
    i = 0
    z = 0
    Q_vector = list()
    while i < rows:
        r = list()
        j = 0
        while j < columns:
            r.append(vector[z])
            z += 1
            j += 1
        Q_vector.append(r)
        i += 1
    return Q_vector

def encode_matrix(Q):
        # Q has to be in matrix format
        if isinstance(Q,Matrix) == False:
            Q = Matrix(Q)
        #gets the number of rows and columns and the elements of the matrix
        rows = Q.shape[0]
        columns = Q.shape[1]
        vector = []
        for q in Q:
            vector.append(q)
        # transform them into strings
        r_str = str(rows)
        c_str = str(columns)
        v_str = str()
        for v in vector:
            v_str = v_str + str(v) + '/'
        # create the message to send to the client
        msg = r_str + '|' + c_str + '|' + v_str
        byte_msg = bytes(msg,'utf-8')
        return byte_msg

#motor = Motor(Port.A)
class Robot():

    def __init__(self,name):
        self.hub = PrimeHub()
        self.color_sensor = ColorSensor(Port.B)
        self.force_sensor = ForceSensor(Port.F)
        self.right_motor = Motor(Port.C)
        self.left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)
        self.moving_motors = DriveBase(self.left_motor, self.right_motor, wheel_diameter=68, axle_track=112)
        self.name = name
        self.angle = 0
        self.moving_motors.settings(80,120,30,90)

    def display_text(self,text):
        if isinstance(text,str) == False:
            text = str(text)
        self.hub.display.text(text)

    def display_arrows(self,action):
        if action == 0:
            self.hub.display.icon(Icon.ARROW_LEFT)
        elif action == 1:
            self.hub.display.icon(Icon.ARROW_RIGHT)
        elif action == 2:
            self.hub.display.icon(Icon.ARROW_DOWN)
        elif action == 3:
            self.hub.display.icon(Icon.ARROW_UP)

    def get_angle(self):
        return self.angle

    def set_angle(self,a):
        self.angle = a

    def move_up(self,dis):
        if self.angle == 0:
            self.moving_motors.straight(-dis)
            self.moving_motors.stop()
            wait(500)
        elif self.angle == 90:
            self.moving_motors.turn(-90)
            self.moving_motors.stop()
            wait(500)
            self.moving_motors.straight(-dis)
            self.moving_motors.stop()
            wait(500)
            self.angle = 0
        elif self.angle == -90:
            self.moving_motors.turn(90)
            self.moving_motors.stop()
            wait(500)
            self.moving_motors.straight(-dis)
            self.moving_motors.stop()
            wait(500)
            self.angle = 0

    def move_down(self,dis):
        if self.angle == 0:
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(500)
        elif self.angle == 90:
            self.moving_motors.turn(-90)
            self.moving_motors.stop()
            wait(500)
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(500)
            self.angle = 0
        elif self.angle == -90:
            self.moving_motors.turn(90)
            self.moving_motors.stop()
            wait(500)
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(500)
            self.angle = 0

    def move_right(self,dis):
        if self.angle == 0:
            self.moving_motors.turn(90)
            self.moving_motors.stop()
            wait(500)
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(500)
            self.angle += 90
        elif self.angle == 90:
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(500)
        elif self.angle == -90:
            self.moving_motors.turn(180)
            #self.moving_motors.turn(77)
            self.moving_motors.stop()
            wait(500)
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(500)
            self.angle += 180

    def move_left(self,dis):
        if self.angle == 0:
            self.moving_motors.turn(-90)
            self.moving_motors.stop()
            wait(500)
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(500)
            self.angle += -90
        elif self.angle == -90:
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(500)
        elif self.angle == 90:
            self.moving_motors.turn(-180)
            #self.moving_motors.turn(-77)
            self.moving_motors.stop()
            wait(500)
            self.moving_motors.straight(dis)
            self.moving_motors.stop()
            wait(500)
            self.angle += -180

    def idle(self):
        ack = None
        try:
            ack = stdin.buffer.read(3)
            #ack = repr(temp)
            self.hub.display.text(ack)
        except Exception as e:
            return True
        if ack != None:
            return False
        else:
            return True

class Environment():
    def __init__(self):
        self.hub = Robot('Spiky')
        self.start = [0,0]
        self.position = self.start
        self.nRows = 5
        self.nCols = 5

    def eps_greedy(self, s, Q, eps):
        actions = [0,1,2,3]
        if urandom.random() <= eps:
            a = urandom.choice(actions)
        else:
            Q_s = Q[s]
            a = Q_s.index(max(Q_s))
        return int(a)

    def do_move(self,a,d):
        if a == 0:
            self.hub.move_up(d)
        elif a == 1:
            self.hub.move_down(d)
        elif a == 2:
            self.hub.move_right(d)
        elif a == 3:
            self.hub.move_left(d)

    def undo_move(self,a,d):
        if a == 0:
            self.hub.move_down(d)
        elif a ==1:
            self.hub.move_up(d)
        elif a == 2:
            self.hub.move_left(d)
        elif a == 3:
            self.hub.move_right(d)

    def comeback_function(self,d):
        p = self.hub.get_position()
        up = p[0]*d
        right = p[1]*d
        self.hub.move_up_advanced(up)
        self.hub.move_right_advanced(right)
        self.hub.moving_motors.turn(-90)
        self.hub.angle = 0

    def set_position(self,pos):
        self.position = pos
    #def create_grid(self):
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
    def tuple_to_state(self,t):
        s = t[0]*self.nCols + t[1]
        return s
    def transition_model(self,a):
        # action encoding:
        # 0: up
        # 1: down
        # 2: right
        # 3: left

        s_tuple = self.hub.get_position()
        s = self.hub.tuple_to_state(s_tuple)
        self.update_position(a)
        s_prime = self.tuple_to_state(self.get_position())
        if s_prime[0] < 0 or s_prime[1] < 0 or s_prime[0] >= self.nRows or s_prime[1] >= self.nCols:
            s_prime = s
            self.set_position(s_tuple)
        color = self.hub.color_sensor.color(True)
        if color == Color.BLUE:
            reward = -0.1
        elif color == Color.RED or color == Color.NONE:  # set the second color according to the borders:
            reward = -1
        elif color == Color.GREEN:
            reward = 1
        return s_prime,reward

def main():

    env = Environment()
    # 1. ack iniziale
    stdout.buffer.write(b'ack')

    while True:
        # 2. starting function implementation
        # wait until something happens

        while env.hub.force_sensor.pressed() == False:
            env.hub.display.text('S')
        stdout.buffer.write(b'start')
        # primo acknowlegement di sincronizzazione
        flag = True
        while flag:
            env.hub.hub.light.on(Color.MAGENTA)
            flag = env.hub.idle()
        # il robot è pronto a ricevere Q
        stdout.buffer.write(b'ack')
        # ack per Q... resta in attesa
        flag = True
        while flag:
            env.hub.hub.light.on(Color.RED)
            flag = env.hub.idle()
        # read Q
        env.hub.hub.light.on(Color.CYAN)
        x = stdin.buffer.read(1)
        env.hub.hub.display.text(x) # lettura lunghezza buffer
        n = int(x)
        q_msg = ''
        for j in range(0,n):
            env.hub.hub.display.text(str(j))
            stdout.buffer.write(b'ack')
            temp = decripting_function()
            q_msg = q_msg + temp
            stdout.buffer.write(b'ack')
        #hub.display.text(q_msg)
        env.hub.hub.light.on(Color.ORANGE)
        Q = decode_matrix(q_msg)
        if Q != None:
            env.hub.hub.light.on(Color.GREEN)
            env.hub.hub.display.text('Q')
            stdout.buffer.write(b'ack')

        # ack per eps
        flag = True
        while flag:
            env.hub.hub.light.on(Color.RED)
            flag = env.hub.idle()
            wait(1000)
        # read eps
        eps_msg = decripting_function()
        eps = float(eps_msg)
        if eps != None:
            env.hub.hub.light.on(Color.GREEN)
            env.hub.hub.display.text('s')
            stdout.buffer.write(b'ack')

        env.hub.hub.display.text('Cook..')
        # DA SISTEMARE
        # 3. now we create a matrix that saves the three action taken and the liked data to send bacl to the pc
        results = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        s = 0

        a = env.eps_greedy(s,Q,1.)

        for i in range(0,8):
            env.hub.hub.display.text(str(s))
            wait(500)
            env.hub.hub.display.text(str(a))
            wait(500)
            env.do_move(a,200)
            s_prime, r = env.transition_model(s,a)
            results[i] = [s,a,r]
            a_prime = env.eps_greedy(s_prime, Q, eps)
            if s_prime == s:
                env.undo_move(a,200)
            # next iteration
            s = s_prime
            a = a_prime
        env.comeback_function(200)
        stdout.buffer.write(b'ack')
        msg_matrix = encode_matrix(results)
        # 4. send the results
        env.hub.hub.display.text('SR')
        stdout.buffer.write(msg_matrix)
        flag = True
        # 5.wait for computer computation
        while flag:
            env.hub.hub.light.on(Color.BLACK)
            flag = env.idle()
        env.hub.hub.light.on(Color.GREEN)