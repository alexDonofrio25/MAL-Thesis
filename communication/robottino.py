from pybricks.pupdevices import Motor, ForceSensor
from pybricks.hubs import PrimeHub
from pybricks.parameters import Port, Icon, Color
from pybricks.tools import wait
from pybricks.geometry import Matrix
import ustruct

# Standard MicroPython modules
from usys import stdin, stdout
from uselect import poll

#motor = Motor(Port.A)
hub = PrimeHub()
force_sensor = ForceSensor(Port.B)

# Optional: Register stdin for polling. This allows
# you to wait for incoming data without blocking.
keyboard = poll()
keyboard.register(stdin)


def decripting_function():
    msg = ''
    flag = True
    while flag:
        cmd = stdin.buffer.read(5)
        t = repr(cmd)
        v = t.replace('b','')
        temp = v.replace("'",'')
        temp1 = ''
        for t in temp:
            if t == '*': # special char that indicates the end of a message
                flag = False
            else:
                temp1 = temp1 + t
        msg = msg + temp1
    return msg

def decoding_function():
    msg = ''
    flag = True
    while flag:
        cmd = stdin.buffer.read(5)
        t = repr(cmd)
        v = t.replace('b','')
        temp = v.replace("'",'')
        temp1 = ''
        for t in temp:
            if t == '*': # special char that indicates the end of a message
                flag = False
            else:
                temp1 = temp1 + t
        msg = msg + temp1
    return msg

# 1. send an ack to communicate the program is started
stdout.buffer.write(b"ack")

# 2. starting function implementation
    # wait until something happens
check = force_sensor.pressed()
while check == False:
    check = force_sensor.pressed()
    hub.display.text('S')
stdout.buffer.write(b"start")

while True:

    # wait for data
    data = None
    while data == None:
        try:
            hub.light.on(Color.RED)
            data = decripting_function()
        except Exception as e:
            hub.light.on(Color.ORANGE)
    hub.display.text(data)

    break
hub.display.text('addio ragazzi')
stdout.buffer.write(b'routine')




