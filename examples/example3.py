import sys
sys.path.insert(0,'..')
from pyminfunc import minFunc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# silly function
def myf(x):
     sinx1 = np.sin(x[1])
     cosx1 = np.cos(x[1])
     f = x[0]**2 * sinx1**2 + (x[1]-10)**2 + (x[0]-10)**2
     g = np.array([2*x[0] * sinx1**2 + (x[0]-10),
                   x[0]**2 * sinx1 * cosx1 + (x[1]-10)])
     return f,g

class recordevals:
    def __init__(self,f):
        self.f = f
        self.xs = None

    def __call__(self,x):
        if self.xs is None:
            self.xs = x[None,:]
        else:
            self.xs = np.vstack((self.xs,x[None,:]))
        return self.f(x)


# plot fn's contours
xs = np.linspace(-5,25,100)
ys = np.linspace(-5,35,100)
xx,yy = np.meshgrid(xs,ys)
ff = myf(np.array([xx,yy]))[0]
plt.contour(xs,ys,ff,cmap=cm.gray)

# find minimum and plot
x0 = np.array([0.5,-0.25])
myfrec = recordevals(myf)
# now switch method to steepest descent
xmin = minFunc(myfrec,x0,{'display':'off','noutputs':1,'method':'sd'})
pts = myfrec.xs
plt.scatter(pts[:,0],pts[:,1],20,-np.arange(pts.shape[0]),cmap=cm.gist_heat)
plt.scatter([xmin[0]],[xmin[1]],50,'k',marker='*')
# show path
plt.show()
