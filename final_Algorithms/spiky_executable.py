from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, ColorSensor, UltrasonicSensor
from pybricks.parameters import Color, Direction, Port, Stop, Icon
from pybricks.robotics import DriveBase
from pybricks.tools import wait
from usys import stdin, stdout

class Robot():

    def __init__(self):
        self.hub = PrimeHub()
        self.color_sensor = ColorSensor(Port.B)
        self.distance_sensor = UltrasonicSensor(Port.A)
        self.right_motor = Motor(Port.C)
        self.left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)
        self.moving_motors = DriveBase(self.left_motor, self.right_motor, wheel_diameter=88, axle_track=144.75)
        self.angle = 0
        self.moving_motors.settings(800,450,720,360)
        self.down = 0
        self.left = 0

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
        while self.moving_motors.state()[2] > teta -deg:
            if self.moving_motors.state()[2] > teta -0.8*deg :
                self.moving_motors.turn(-20)
            else:
                self.moving_motors.turn(-2)
        self.moving_motors.stop()

    def setSpeed(self,speed,acc):
        self.moving_motors.settings(speed,acc,720,360)

    def resetSettings(self):
        self.moving_motors.settings(800,450,720,360)

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
            self.hub.display.icon(Icon.ARROW_UP)
        elif action == 3:
            self.hub.display.icon(Icon.ARROW_DOWN)

    def get_angle(self):
        return self.angle

    def set_angle(self,a):
        self.angle = a

    def move_up(self,dis):
        self.display_arrows(0)
        teta = self.moving_motors.state()[2]
        if self.angle == 0:
            self.moving_motors.turn(-teta)
            wait(100)
            self.moving_motors.straight(-dis,Stop.BRAKE)
        elif self.angle == 90:
            self.turn_left(teta)
            wait(100)
            self.moving_motors.straight(-dis,Stop.BRAKE)
        elif self.angle == -90:
            self.turn_right(-teta)
            wait(100)
            self.moving_motors.straight(-dis,Stop.BRAKE)
        self.angle = 0

    def move_down(self,dis):
        self.display_arrows(1)
        teta = self.moving_motors.state()[2]
        if self.angle == 0:
            self.moving_motors.turn(-teta)
            wait(100)
            self.moving_motors.straight(dis,Stop.BRAKE)
        elif self.angle == 90:
            self.turn_left(teta)
            wait(100)
            self.moving_motors.straight(dis,Stop.BRAKE)
        elif self.angle == -90:
            self.turn_right(-teta)
            wait(100)
            self.moving_motors.straight(dis,Stop.BRAKE)
        self.angle = 0

    def move_right(self, dis):
        self.display_arrows(3)
        teta = self.moving_motors.state()[2]
        if self.angle == 0:
            self.turn_right(90 - teta)
            wait(100)
            self.moving_motors.straight(dis,Stop.BRAKE)
            self.angle = 90
        elif self.angle == 90:
            self.moving_motors.turn(90-teta)
            wait(100)
            self.moving_motors.straight(dis,Stop.BRAKE)
        elif self.angle == -90:
            self.moving_motors.turn(-90-teta)
            wait(100)
            self.moving_motors.straight(-dis,Stop.BRAKE)

    def move_left(self, dis):
        self.display_arrows(2)
        teta = self.moving_motors.state()[2]
        if self.angle == 0:
            self.turn_left(90 - teta)
            wait(100)
            self.moving_motors.straight(dis,Stop.BRAKE)
            self.angle = -90
        elif self.angle == 90:
            self.moving_motors.turn(90-teta)
            wait(100)
            self.moving_motors.straight(-dis,Stop.BRAKE)
        elif self.angle == -90:
            self.moving_motors.turn(-90-teta)
            wait(100)
            self.moving_motors.straight(dis,Stop.BRAKE)

    def backInTheHub(self):
        self.hub.display.icon(Icon.HAPPY)
        self.hub.light.on(Color.MAGENTA)
        self.setSpeed(250,180)
        #self.turn_left(180)
        i = 1
        self.moving_motors.reset()
        self.moving_motors.straight(200)
        dis = self.distance_sensor.distance()
        while dis > 39:
            if i > 2.1 and i%2 == 0:
                self.moving_motors.curve(150,5*i)
                self.moving_motors.curve(150,-10*i)
            else:
                self.moving_motors.curve(150,-5*i)
                self.moving_motors.curve(150,10*i)
            dis = self.distance_sensor.distance()
            if dis == 2000:
                break
            if i > 2 and i <= 4:
                self.moving_motors.straight(-80)
            dis = self.distance_sensor.distance()
            if dis == 2000 or i == 4.5:
                break
            i+=0.5
        self.setSpeed(350,280)
        self.moving_motors.straight(200)
        self.moving_motors.stop()
        self.moving_motors.reset()

    def comeback_function(self,d):
        #self.hub.moving_motors.settings(200,100,720,360)
        self.hub.light.on(Color.MAGENTA)
        up = self.down*d
        if up != 0:
            self.move_up(up)
        right = self.left*d
        if right != 0:
            self.move_right(right)
        if self.angle == 0:
            self.turn_left(180)
        elif self.angle == 90:
            self.turn_right(90)
        else:
            self.turn_left(90)
        self.angle = 0
        self.down = 0
        self.left = 0

def idle(robot):
        ack = None
        try:
            ack = stdin.buffer.read(4)   #robot1 waits 'ack1' and robot2 waits 'ack2'
            robot.display.icon(Icon.TRUE)
        except Exception as e:
            return True
        if ack != None:
            return False
        else:
            return True

# the robot wait until an action is sent to it
def actionLoop():
    robot = Robot()
    # 1. ack iniziale
    stdout.buffer.write(b'ack')
    # learning parameters
    while True:
        robot.hub.light.on(Color.RED)
        idle(robot)
        robot.hub.light.on(Color.GREEN)
        action = stdin.buffer.read(1)
        if action == b'0':
            robot.move_up(240)
            robot.down -= 1
        elif action == b'1':
            robot.move_down(240)
            robot.down += 1
        elif action == b'2':
            robot.move_right(240)
            robot.left -= 1
        elif action == b'3':
            robot.move_left(240)
            robot.left += 1
        elif action == b'4':
            color = robot.color_sensor.color(True)
            if color == Color.BLUE:
                stdout.buffer.write(b'b')
            elif color == Color.RED:
                stdout.buffer.write(b'r')
            elif color == Color.GREEN:
                stdout.buffer.write(b'g')
        elif action == b'5':
            # go out of the box
            robot.move_up(280)
            robot.turn_right(180)
            robot.resetSettings()
            robot.moving_motors.reset()
            robot.angle = 0
        elif action == b'6':
            # get in the hub
            robot.backInTheHub()
        elif action == b'7':
            # get in the hub
            robot.comeback_function(240)
        elif action == b'9':
            break
        else:
            robot.hub.display.icon(Icon.SAD)
        stdout.buffer.write(b'finish')
    stdout.buffer.write(b'END')

actionLoop()

