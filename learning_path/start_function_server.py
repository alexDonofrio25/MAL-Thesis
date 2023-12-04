from pybricks.pupdevices import Motor, UltrasonicSensor
from pybricks.robotics import DriveBase
from pybricks.hubs import PrimeHub
from pybricks.parameters import Port, Direction, Icon
from pybricks.tools import wait, StopWatch

# Standard MicroPython modules
from usys import stdin, stdout
from uselect import poll

# Optional: Register stdin for polling. This allows
# you to wait for incoming data without blocking.
#keyboard = poll()
hub = PrimeHub()
distance_sensor = UltrasonicSensor(Port.F)
right_motor = Motor(Port.C)
left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)
moving_motors = DriveBase(left_motor, right_motor, wheel_diameter=56, axle_track=112)
#keyboard.register(distance_sensor)
# send this ack to let the client know the program is started
stdout.buffer.write(b'ack')
while distance_sensor.distance() > 50:
    hub.display.icon(Icon.HEART)
stdout.buffer.write(b'start')

while True:
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
            moving_motors.straight(50)
        elif cmd == b'turnR':
            #do something
            stdout.buffer.write(cmd)
            moving_motors.turn(90)
        elif cmd == b'light':
            #do something
            stdout.buffer.write(cmd)
            hub.speaker.beep(40)