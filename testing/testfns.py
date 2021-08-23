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

def colville1(x):
    return 100*(x[0]**2 - x[1])**2 + (x[0]-1)**2 + (x[2]-1)**2 + 90*(x[2]**2 - x[3])**2 + 10.1*((x[1]-1)**2 + (x[3]-1)**2) + 19.8*(x[1]-1)*(x[3]-1)

def colville2(x):
    f = colville1(x)
    g = np.array([400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2,
                   -200*x[0]**2 + 220.2*x[1] + 19.8*x[3] - 40,
                   360*x[2]**3 - 360*x[2]*x[3] + 2*x[2] - 2,
                   19.8*x[1] - 180*x[2]**2 + 200.2*x[3] - 40])
    return f,g

def colville3(x):
    f,g = colville2(x)
    H = np.array([[1200*x[0]**2 - 400*x[1] +2, -400*x[0], 0 , 0 ],
                  [-400*x[0], 220.2, 0, 19.8],
                  [0, 0, 1080*x[2]**2 - 360*x[3] + 2, -360*x[2]],
                  [0, 19.8, -360*x[2], 200.2]])
    return f,g,H

def colville4(x):
    f,g,H = colville3(x)
    T = np.zeros((4,4,4))
    T[0,0,0] = 2400*x[0]
    T[0,0,1] = T[0,1,0] = T[1,0,0] = -400
    T[2,2,2] = 2160*x[2]
    T[2,2,3] = T[2,3,2] = T[3,2,2] = -360
    return f,g,H,T



