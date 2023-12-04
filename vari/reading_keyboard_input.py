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

start_time = watch.time()
while True:


    # Optional: Check available input.
    while not keyboard.poll(0):
        hub.display.text('hello')
        # Optional: Do something here.
        wait(10)
    # Read three bytes.
    cmd = stdin.buffer.read(4)
    # Decide what to do based on the command.
    if cmd == b'exit':
        stdout.buffer.write(b"STOP")
        break
    else:
        stdout.buffer.write(cmd)
        #dec = cmd.decode('utf-8')
        hub.display.text(cmd)

    # Send a response.
    #stdout.buffer.write(b"OK")

