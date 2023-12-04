from pybricks.pupdevices import Motor, UltrasonicSensor
from pybricks.robotics import DriveBase
from pybricks.hubs import PrimeHub
from pybricks.parameters import Port, Direction, Icon, Button
from pybricks.tools import wait, StopWatch

# Standard MicroPython modules
from usys import stdin, stdout
from uselect import poll

# Optional: Register stdin for polling. This allows
# you to wait for incoming data without blocking.
#keyboard = poll()
hub = PrimeHub()
#distance_sensor = UltrasonicSensor(Port.F)
#right_motor = Motor(Port.C)
#left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)
#moving_motors = DriveBase(left_motor, right_motor, wheel_diameter=56, axle_track=112)
#keyboard.register(distance_sensor)
# send this ack to let the client know the program is started
stdout.buffer.write(b'ack')

def not_in(collection,element):
    t = 0
    for c in collection:
        if c == element:
            t = 1
    if t == 1:
        return False
    else:
        return True

while True:
    # now wait an action
    while not_in(hub.buttons.pressed(),Button.RIGHT):
        hub.display.icon(Icon.HEART)
    stdout.buffer.write(b'start')

    # Read five bytes from the buffer.
    cmd = stdin.buffer.read(5)
    stdout.buffer.write(cmd)
    # Decide what to do based on the command.
    if cmd == b'exitt':
        stdout.buffer.write(b"STOP")
        break
    else:
        if cmd == b'fwd05':
            #do something
            hub.display.number(1)
        elif cmd == b'turnR':
            #do something
            hub.speaker.play_notes(['A4/4','C4/4'])
            stdout.buffer.write(cmd)
        elif cmd == b'light':
            #do something
            stdout.buffer.write(cmd)
            hub.speaker.beep(40)