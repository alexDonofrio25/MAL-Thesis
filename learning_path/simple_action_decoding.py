from pybricks.pupdevices import Motor
from pybricks.robotics import DriveBase
from pybricks.hubs import PrimeHub
from pybricks.parameters import Port,Direction
from pybricks.tools import wait, StopWatch

# Standard MicroPython modules
from usys import stdin, stdout
from uselect import poll

#motor = Motor(Port.A)
watch = StopWatch()

# Optional: Register stdin for polling. This allows
# you to wait for incoming data without blocking.
keyboard = poll()
keyboard.register(stdin)
hub = PrimeHub()
right_motor = Motor(Port.C)
left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)
moving_motors = DriveBase(left_motor, right_motor, wheel_diameter=56, axle_track=112)


start_time = watch.time()
while True:


    # Optional: Check available input.
    while not keyboard.poll(0):
        hub.display.text('hello')
        # Optional: Do something here.
        wait(10)
    # Read five bytes from the buffer.
    cmd = stdin.buffer.read(5)
    # Decide what to do based on the command.
    if cmd == b'exitt':
        stdout.buffer.write(b"STOP")
        break
    else:
        stdout.buffer.write(cmd)
        if cmd == b'fwd20':
            #do something
            stdout.buffer.write(cmd)
            moving_motors.straight(200)
        elif cmd == b'turnR':
            #do something
            stdout.buffer.write(cmd)
            moving_motors.turn(90)
        elif cmd == b'light':
            #do something
            stdout.buffer.write(cmd)
            hub.speaker.beep(40)
    # Send a response.
    #stdout.buffer.write(b"OK")
