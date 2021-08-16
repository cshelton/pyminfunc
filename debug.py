import numpy as np
import struct

def tohex(v):
    if isinstance(v,np.float64):
        return format(struct.unpack('!Q',struct.pack('!d',v))[0],'016x')
    elif isinstance(v,np.float32):
        return format(struct.unpack('!I',struct.pack('!f',v))[0],'016x')
    elif isinstance(v,np.int):
        return format(struct.unpack('!I',struct.pack('!i',v))[0],'016x')
    else:
        return ''

def debugstr(x):
    if isinstance(x,np.ndarray):
        mid = ' ' if len(x.shape)==1 else '\n'
        return '['+mid.join([debugstr(v) for v in x])+']'
    return '{:.20g} ({:s})'.format(x,tohex(x))
