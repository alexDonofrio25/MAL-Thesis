# Spiky

from pybricks.pupdevices import Motor, ForceSensor, UltrasonicSensor
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
force_sensor = ForceSensor(Port.B)
distance_sensor = UltrasonicSensor(Port.F)
right_motor = Motor(Port.C)
left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)
moving_motors = DriveBase(left_motor, right_motor, wheel_diameter=80, axle_track=112)
moving_motors.settings(80,120,30,90)

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

def decode_matrix(msg):
    if isinstance(msg,bytes):  # check if the input message is a bytearray, if yes, it casts it into a
        msg = str(msg,'utf-8')

    split = msg.rsplit('|')
    rows = int(float(split[0]))
    columns = int(float(split[1]))

    vector = list()

    v_split = split[2].rsplit('/')
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
    return Q_vector

def encode_matrix(Q):
        # Q has to be in matrix format
        if isinstance(Q,Matrix) == False:
            Q = Matrix(Q)
        #gets the number of rows and columns and the elements of the matrix
        rows = Q.shape[0]
        columns = Q.shape[1]
        vector = []
        for q in Q:
            vector.append(q)
        # transform them into strings
        r_str = str(rows)
        c_str = str(columns)
        v_str = str()
        for v in vector:
            v_str = v_str + str(v) + '/'
        # create the message to send to the client
        msg = r_str + '|' + c_str + '|' + v_str
        byte_msg = bytes(msg,'utf-8')
        return byte_msg

def eps_greedy(s, Q, eps, allowed_actions):
    qM = Matrix(Q)
    if urandom.random() <= eps:
        actions = []
        for a in allowed_actions:
            if a == 1:
                actions.append(allowed_actions.index(a))
        a = urandom.choice(actions)
    else:
        Q_s = []
        for i in range(0, qM.shape[1]):
            Q_s.append(qM[s,i])
        Q_s[allowed_actions == 0] = - 1000
        a = max(Q_s)
    return umath.floor(a)

def state_detection():
    dis = umath.floor(distance_sensor.distance())
    print(dis)
    if dis == 400:
        return 0
    elif dis == 300:
        return 1
    elif dis == 200:
        return 2
    elif dis == 100:
        return 3
    elif dis == 0:
        return 4
    else:
        return 0

def transition_model(s,a):
    s_prime = s + a + 1
    if s == 2 and a == 0:
        inst_rew = 1.0
    else:
        inst_rew = -5
    return s_prime, inst_rew

def comeback_function(start):
    d = distance_sensor.distance()
    distance_to_do = start - d
    moving_motors.straight(-distance_to_do)

def send_message(msg):
    m = bytes(msg,'utf-8')
    stdout.buffer.write(m)

def idle():
    ack = None
    try:
        ack = stdin.buffer.read(3)
        #ack = repr(temp)
        hub.display.text(ack)
    except Exception as e:
        return True
    if ack != None:
        return False
    else:
        return True

def bho():
    # ack iniziale
    stdout.buffer.write(b'ack')

    # 2. starting function implementation
        # wait until something happens

    while force_sensor.pressed() == False:
        hub.display.text('S')
    stdout.buffer.write(b'start')
    wait(1000)
    hub.display.icon(Icon.CIRCLE)
    x = stdin.buffer.read(1)
    n = int(x)
    res = ''
    for j in range(0,n):
        res = res + decripting_function()
        stdout.buffer.write(b'ack')
    #hub.display.text(x)
    #stdout.buffer.write(b'ack')
    y = decripting_function()
    stdout.buffer.write(b'exit')
    hub.display.text('bye')

def main():
    # 1. ack iniziale
    stdout.buffer.write(b'ack')

    # 2. starting function implementation
        # wait until something happens

    while force_sensor.pressed() == False:
        hub.display.text('S')
    stdout.buffer.write(b'start')

    while True:
        # primo acknowlegement di sincronizzazione
        flag = True
        while flag:
            hub.light.on(Color.MAGENTA)
            flag = idle()
        # il robot è pronto a ricevere Q
        stdout.buffer.write(b'ack')
        # ack per Q... resta in attesa
        flag = True
        while flag:
            hub.light.on(Color.RED)
            flag = idle()
        # read Q
        hub.light.on(Color.CYAN)
        x = stdin.buffer.read(1)
        hub.display.text(x) # lettura lunghezza buffer
        n = int(x)
        q_msg = ''
        for j in range(0,n):
            hub.display.text(str(j))
            stdout.buffer.write(b'ack')
            temp = decripting_function()
            q_msg = q_msg + temp
            stdout.buffer.write(b'ack')
        #hub.display.text(q_msg)
        hub.light.on(Color.ORANGE)
        Q = decode_matrix(q_msg)
        if Q != None:
            hub.light.on(Color.GREEN)
            hub.display.text('Q')
            stdout.buffer.write(b'ack')

        # ack per eps
        flag = True
        while flag:
            hub.light.on(Color.RED)
            flag = idle()
            wait(1000)
        # read eps
        eps_msg = decripting_function()
        eps = float(eps_msg)
        if eps != None:
            hub.light.on(Color.GREEN)
            hub.display.text('s')
            stdout.buffer.write(b'ack')


        allowed_actions = [[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]]
        hub.display.text('Cook..')
        # DA SISTEMARE
        # 3. now we create a matrix that saves the three action taken and the liked data to send bacl to the pc
        results = [[0,0,0],[0,0,0],[0,0,0]]
        distances = [0,0,0,0,0,0,0]
        while distance_sensor.distance() > 400:
            moving_motors.drive(80,0)
        moving_motors.stop()

        s = 0
        distances[s] = umath.floor(distance_sensor.distance())
        a = eps_greedy(s,Q,1.,allowed_actions[s])

        for i in range(0,3):
            hub.display.text(str(s))
            wait(500)
            hub.display.text(str(a))
            wait(500)
            if a == 0:
                moving_motors.straight(100)
            if a == 1:
                moving_motors.straight(200)
            if a == 2:
                moving_motors.straight(300)
            s_prime, r = transition_model(s,a)
            results[i] = [s,a,r]
            if s_prime == 5:
                break
            a_prime = eps_greedy(s_prime, Q, eps, allowed_actions[s_prime])
            # next iteration
            s = s_prime
            a = a_prime
        comeback_function(distances[0])
        stdout.buffer.write(b'ack')
        msg_matrix = encode_matrix(results)
        # 4. send the results
        hub.display.text('SR')
        stdout.buffer.write(msg_matrix)
        flag = True
        # 5.wait for computer computation
        while flag:
            hub.light.on(Color.BLACK)
            flag = idle()
        hub.light.on(Color.GREEN)

main()
