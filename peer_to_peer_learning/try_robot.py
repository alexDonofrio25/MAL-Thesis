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

#motor = Motor(Port.A)
hub = PrimeHub()
right_motor = Motor(Port.C)
left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)


def idle():
    ack = None
    try:
        ack = stdin.buffer.read(3)
        #ack = repr(temp)
        #hub.display.text(ack)
    except Exception as e:
        return True
    if ack != None:
        return False
    else:
        return True

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
# 1. ack iniziale
stdout.buffer.write(b'ack')

# primo acknowlegement di sincronizzazione
flag = True
while flag:
    hub.light.on(Color.MAGENTA)
    flag = idle()
# il robot è pronto a ricevere at
stdout.buffer.write(b'ack')
flag = True
while flag:
    hub.light.on(Color.RED)
    flag = idle()
    wait(1000)
# read eps
at_msg = decripting_function()
at = float(at_msg)
x = 0
if at != None:
    hub.light.on(Color.GREEN)
    #hub.display.text('at')
moving_motors = DriveBase(left_motor, right_motor, wheel_diameter=80, axle_track=at)
moving_motors.settings(80,120,720,360)
moving_motors.turn(90)
teta = moving_motors.state()[2]
t = str(teta)
hub.display.text(t)
moving_motors.reset()
if teta == 90:
    x = 1
else:
    x = 0
x_str = str(x)
x_msg = bytes(x_str,'utf-8')
stdout.buffer.write(x_msg)
