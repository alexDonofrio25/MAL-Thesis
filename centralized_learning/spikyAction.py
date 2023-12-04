from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, ColorSensor, UltrasonicSensor, ForceSensor
from pybricks.parameters import Button, Color, Direction, Port, Side, Stop, Icon
from pybricks.robotics import DriveBase
from pybricks.tools import wait, StopWatch
from usys import stdin, stdout

hub = PrimeHub()
right_motor = Motor(Port.C)
left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)
moving_motors = DriveBase(left_motor, right_motor, wheel_diameter=88, axle_track=144.75)
moving_motors.settings(300,180,360,180)
moving_motors.reset()

def idle():
        ack = None
        try:
            ack = stdin.buffer.read(4)   #robot1 waits 'ack1' and robot2 waits 'ack2'
            hub.display.icon(Icon.TRUE)
        except Exception as e:
            return True
        if ack != None:
            return False
        else:
            return True

def doAction(a,d):
    deg = moving_motors.state()[2]
    if a == 0:
        moving_motors.turn(-deg) # si aggiusta
        moving_motors.straight(d) # va avanti
        moving_motors.stop()
    elif a == 1:
        moving_motors.turn(-deg) # si aggiusta
        moving_motors.straight(-d)
        moving_motors.stop()

def comeback(dis):
    hub.light.on(Color.VIOLET)
    if dis != 0:
        x = dis*300
        doAction(1,x)


def actionLoop():
    # 1. ack iniziale
    stdout.buffer.write(b'ack')
    # learning parameters
    M = 30
    m = 1
    k = 2 # length of the episode

    while m<M:
        dis = 0
        hub.display.text(str(m))
        #second = 0
        for i in range(0,k):
            hub.light.on(Color.RED)
            idle()
            hub.light.on(Color.GREEN)
            action = stdin.buffer.read(1)
            if action == b'0':
                doAction(0,300)
                dis += 1
            elif action == b'1':
                doAction(1,300)
                dis -= 1
            else:
                hub.display.icon(Icon.SAD)
            stdout.buffer.write(b'finish')
        comeback(dis)
        stdout.buffer.write(b'end')
        m += 1
    hub.display.text('Finish')

actionLoop()

