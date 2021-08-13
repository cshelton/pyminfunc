from minFunc import *
import numpy as np
from scipy.optimize import rosen,rosen_der,rosen_hess

def rosenbrock1(x):
    return rosen(x)

def rosenbrock2(x):
    return rosen(x), rosen_der(x)

def rosenbrock3(x):
    return rosen(x), rosen_der(x), rosen_hess(x)



