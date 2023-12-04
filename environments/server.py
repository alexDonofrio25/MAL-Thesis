# in this case the server is intended as the robot Spiky itself

from pybricks.pupdevices import Motor, UltrasonicSensor
from pybricks.robotics import DriveBase
from pybricks.hubs import PrimeHub
from pybricks.parameters import Port, Direction, Icon
from pybricks.tools import wait, StopWatch
from pybricks.geometry import Matrix

# Standard MicroPython modules
from usys import stdin, stdout
from uselect import poll
from umath import fmod, floor




class Server():

    def __init__(self):
        self.hub = PrimeHub()
        self.distance_sensor = UltrasonicSensor(Port.F)
        self.right_motor = Motor(Port.C)
        self.left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)
        self.moving_motors = DriveBase(self.left_motor, self.right_motor, wheel_diameter=56, axle_track=112)


    def move_foreward(self,distance):
        # this function makes the hub move foreward for a specified distance in mm
        self.moving_motors.straight(distance)

    def starting_function(self):
        # send this ack to let the client know the program is started
        stdout.buffer.write(b'ack')

        while self.distance_sensor.distance() > 50:
            self.hub.display.icon(Icon.HEART)

        # tell the client that the hub is ready to receive data
        stdout.buffer.write(b'start')

    def listening(self):
        input = stdin.buffer.read()
        input = input.decode()
        return input

    def measure_distance(self):
        #measure the distance of the hub from an obstacle in mm
        distance = self.distance_sensor.distance()
        return distance

    def set_state(self):
        d = self.measure_distance()
        current_state = (400 - d)/100
        return floor(current_state)

    def compute_reward(self, action):
        s = self.set_state()
        if s == 2 and action == 0:
            reward = 1
        else:
            distance_done = float((action + 1)*10)
            distance_measured =  self.hub.distance_sensor.distance()/10
            inst_rew = -(distance_measured/distance_done)
        return reward

    def encode_matrix(self, Q):
        # Q has to be in matrix format
        if isinstance(Q,Matrix) == False:
            Q = Matrix(Q)
        #gets the number of rows and columns and the elements of the matrix
        rows = Q.shape[0]
        columns = Q.shape[1]
        vector = []
        for q in self.Q:
            vector.append(q)
        # transform them into strings
        r_str = str(rows)
        c_str = str(columns)
        v_str = str()
        for v in vector:
            v_str = v_str + str(v) + '-'
        # create the message to send to the client
        msg = r_str + '|' + c_str + '|' + v_str
        byte_msg = bytes(msg)
        return byte_msg

    def decode_matrix(self, msg):
        if isinstance(msg,bytes):  # check if the input message is a bytearray, if yes, it casts it into a
            msg = str(msg,'utf-8')
        split = msg.rsplit('|')
        rows = float(split[0])
        columns = float(split[1])
        vector = list()
        v_split = split[2].rsplit('-')
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
        Q = Matrix(Q_vector)
        return Q_vector, Q

    def wait_for_ack(self):
        ack = None
        while ack == None:
            ack = stdin.buffer.read(3)
        if ack == b'ack':
            stdout.buffer.write(ack)




