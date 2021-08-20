import numpy as np
from scipy.optimize import rosen,rosen_der,rosen_hess

def rosenbrock1(x):
    return rosen(x)

def rosenbrock2(x):
    return rosen(x), rosen_der(x)

def rosenbrock3(x):
    return rosen(x), rosen_der(x), rosen_hess(x)

def rosenbrock4(x):
    (f,g,H) = rosenbrock3(x)
    n = x.shape[0]
    T = np.zeros((n,n,n))
    di = np.diag_indices(n,3)
    T[di[0][0:-1],di[1][0:-1],di[2][0:-1]] = 2400*x[0:-1]
    T[di[0][0:-1]+1,di[1][0:-1],di[2][0:-1]] = -400
    T[di[0][0:-1],di[1][0:-1]+1,di[2][0:-1]] = -400
    T[di[0][0:-1],di[1][0:-1],di[2][0:-1]+1] = -400
    return (f,g,H,T)
