# Version 0.9
# Aug 21, 2021
# by Christian Shelton
# based on code by Mark Schmidt

import numpy as np
#from debug import *


def autoHv(v,x,g,useComplex,funObj,*args):
# comments from original Matlab
#% [Hv] = autoHv(v,x,g,useComplex,funObj,varargin)
#%
#% Numerically compute Hessian-vector product H*v of funObj(x,varargin{:})
#%  based on gradient values

    if useComplex:
        mu = 1e-150j
    else:
        mu = 2*np.sqrt(1e-12)*(1+np.linalg.norm(x))/np.linalg.norm(v)
    funval = funObj(x + v*mu,*args)
    finDif = funval[1]
    #np.set_printoptions(precision=15)
    #print('----')
    #print(debugstr(v))
    #print(debugstr(x))
    #print(debugstr(g))
    #print(debugstr(np.linalg.norm(x)))
    #print(debugstr(np.linalg.norm(v)))
    #print(debugstr(mu))
    #print(debugstr(finDif))
    #print(debugstr((finDif-g)/mu))
    return (finDif-g)/mu

