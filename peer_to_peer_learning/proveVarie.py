def encode_matrix(Q):
        nr = len(Q)
        nc = len(Q[0])
        v = []
        for i in range(0,nr):
            for j in range(0,nc):
                v.append(Q[i][j])
        v_str = ''
        for el in v:
            els = str(el)
            v_str = v_str + els + '-'
        nrs = str(len(Q))
        ncs = str(len(Q[0]))
        msg = nrs + '|' + ncs + '|' + v_str
        # write the encoding function from computer to bytearray
        # Q has to be in matrix format
        byte_msg = bytes(msg,'utf-8')
        return byte_msg

def decode_matrix(msg):
    msg_split = msg.rsplit('|')
    nr = int(msg_split[0])
    nc = int(msg_split[1])
    v_string = msg_split[2]
    v = v_string.rsplit('-')
    Q = []
    for i in range(0,nr):
        q = []
        for j in range(0,nc):
            q.append(v[j+i*nc])
        Q.append(q)
    return Q


Q = [[0.0,0.0,0.0], [0.0,0.0,0.0],[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]]
msg = encode_matrix(Q)
msg_string = str(msg,'utf-8')
q = decode_matrix(msg_string)
print(q)