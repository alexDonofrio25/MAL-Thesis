from pybricks.pupdevices import Motor
from pybricks.hubs import PrimeHub
from pybricks.parameters import Port
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
x = 0
start_time = watch.time()
while x == 0:

    # Optional: Check available input.
    while not keyboard.poll(0):
        #hub.display.number(6)
        # Optional: Do something here.
        wait(10)

    # Read three bytes.
    cmd = stdin.buffer.read(3)

    # Decide what to do based on the command.
    if cmd == b"fwd":
        #motor.dc(50)
        hub.display.number(1)
        stdout.buffer.write(cmd)
    elif cmd == b"rev":
        #motor.dc(-50)
        hub.display.number(2)
        stdout.buffer.write(cmd)
    elif cmd == b"bye":
        stdout.buffer.write(b"STOP")
        break
    else:
        #motor.stop()
        x = 1
        stdout.buffer.write(b"STOP")

    # Send a response.






