import numpy as np


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
    return (finDif-g)/mu

