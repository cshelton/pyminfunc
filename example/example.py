from minFunc import minFunc
import numpy as np
import matplotlib.pyplot as plt

def myf(x):
     sinx1 = np.sin(x[1])
     cosx1 = np.cos(x[1])
     f = x[0]**2 * sinx1**2 + (x[1]-10)**2 + (x[0]-10)**2
     g = np.array([2*x[0] * sinx1**2 + (x[0]-10),
                   x[0]**2 * sinx1 * cosx1 + (x[1]-10)])
     return f,g

x0 = np.array([0.5,-0.25])

xs = np.linspace(-15,15,100)
ys = np.linspace(-15,15,100)
xx,yy = np.meshgrid(xs,ys)
ff = myf(np.array([xx,yy]))[0]
plt.contour(xs,ys,ff)
xmin = minFunc(myf,np.array([0,0]),{'display':'off'})[0]
plt.scatter([xmin[0]],[xmin[1]],3,'k')
plt.show()






