# Version 0.9
# Aug 21, 2021
# by Christian Shelton
# based on code by Mark Schmidt

import numpy as np
import scipy.linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu
import scipy.sparse.linalg as sla
#from .debug import *

def isLegal(v):
    if not isinstance(v,np.ndarray): v = np.array(v)
    return not(np.any(np.isinf(v)) or np.any(np.isnan(v)) or np.any(np.iscomplex(v)))

def trisolve2(R,x):
    return la.solve_triangular(R,la.solve_triangular(R,x,trans=1,check_finite=False),check_finite=False)

def precondTriu(r,L):
    #return sla.spsolve_triangular(L.T,sla.spsolve_triangular(L,r,lower=False),lower=False)
    return L(r)

def precondDiag(r,D):
    return D*r

def precondTriuDiag(r,U,D):
    return la.solve_triangular(U,D*la.solve_triangular(U,r,trans=1,check_finite=False),check_finite=False)

def ichol(H,droptol,rule=None):
    import ilupp
    ## a real hack to get a Cholesky from an LU
    #if rule is None:
    #    U = spilu(csc_matrix(H),drop_tol=droptol).U.todense()
    #else:
    #    U = spilu(csc_matrix(H),drop_tol=droptol,drop_rule=rule).U.todense()
    #print(U)
    #D = np.diag(np.sqrt(np.abs(np.diag(U))))
    #R = la.solve_triangular(D,U)
    #return R
    #return ilupp.icholt(csc_matrix(H),H.shape[0],droptol)
    return ilupp.ICholTPreconditioner(csc_matrix(H),H.shape[0],droptol)

def WolfeLineSearch(x,t,d,f,g,gtd,c1,c2,LS_interp,LS_multi,maxLS,progTol,dprint,doPlot,saveHessianComp,useH,funObj,*args):
# from original Matlab code:
#
# Bracketing Line Search to Satisfy Wolfe Conditions
#
# Inputs:
#   x: starting location
#   t: initial step size
#   d: descent direction
#   f: function value at starting location
#   g: gradient at starting location
#   gtd: directional derivative at starting location
#   c1: sufficient decrease parameter
#   c2: curvature parameter
#   debug: display debugging information
#   LS_interp: type of interpolation
#   maxLS: maximum number of iterations
#   progTol: minimum allowable step length
#   doPlot: do a graphical display of interpolation
#   funObj: objective function
#   varargin: parameters of objective function
#
# Outputs:
#   t: step length
#   f_new: function value at x+t*d
#   g_new: gradient value at x+t*d
#   funEvals: number function evaluations performed by line search
#   H: Hessian at initial guess (only computed if requested

    # Evaluate the Objective and Gradient at the Initial Step
    if useH:
        (f_new,g_new,H) = funObj(x+t*d,*args)[0:3]
    else:
        (f_new,g_new) = funObj(x+t*d,*args)[0:2]
    funEvals = 1
    gtd_new = g_new.T@d

    # Bracket an Interval continaing a point satisfying the Wolfe Criteria
    LSiter = 0
    t_prev = 0
    f_prev = f
    g_prev = g
    gtd_prev = gtd
    nrmD = np.max(np.abs(d))
    done = 0

    while LSiter < maxLS:
        ##Bracketing Phase
        if not isLegal(f_new) or not isLegal(g_new):
            dprint('Extrapolated into illegal region, switching to Armijo line-search')
            t = (t+t_prev)/2
            # Do Armijo
            if useH:
                (t,x_new,f_new,g_new,armijoFunEvals,H) = ArmijoBacktrack(x,t,d,f,f,g,gtd,c1,LS_interp,LS_multi,progTol,dprint,doPlot,saveHessianComp,True,funObj,*args)
                funEvals = funEvals + armijoFunEvals
                return (t,f_new,g_new,funEvals,H)
            else:
                (t,x_new,f_new,g_new,armijoFunEvals) = ArmijoBacktrack(x,t,d,f,f,g,gtd,c1,LS_interp,LS_multi,progTol,dprint,doPlot,saveHessianComp,False,funObj,*args)
                funEvals = funEvals + armijoFunEvals
                return (t,f_new,g_new,funEvals)

        if f_new > f+c1*t*gtd or (LSiter > 1 and f_new >= f_prev):
            bracket = [t_prev,t]
            bracketFval = [f_prev,f_new]
            bracketGval = [g_prev,g_new]
            break
        elif np.abs(gtd_new) <= -c2*gtd:
            bracket = [t]
            bracketFval = [f_new]
            bracketGval = [g_new]
            done = 1
            break
        elif gtd_new >= 0:
            bracket = [t_prev,t]
            bracketFval = [f_prev,f_new]
            bracketGval = [g_prev,g_new]
            break

        temp = t_prev
        t_prev = t
        minStep = t + 0.01*(t-temp)
        maxStep = t*10
        if LS_interp <= 1:
            dprint('Extending Braket') # misspelled to match original
            t = maxStep
        elif LS_interp == 2:
            dprint('Cubic Extrapolation')
            t = polyinterp([temp,t],[f_prev,f_new],[gtd_prev,gtd_new],doPlot,minStep,maxStep)
        elif LS_interp==3:
            t = mixedExtrap([temp,t],[f_prev,f_new],[gtd_prev,gtd_new],minStep,maxStep,dprint,doPlot)

        f_prev = f_new
        g_prev = g_new
        gtd_prev = gtd_new
        if not saveHessianComp and useH:
            (f_new,g_new,H) = funObj(x+t*d,*args)[0:3]
        else:
            (f_new,g_new) = funObj(x+t*d,*args)[0:2]
        funEvals += 1
        gtd_new = g_new.T@d
        LSiter += 1

    if LSiter == maxLS:
        bracket = [0,t]
        bracketFval = [f,f_new]
        bracketGval = [g,g_new]

    # Zoom Phase

    # We now either have a point satisfying the criteria, or a bracket
    # surrounding a point satisfying the criteria
    # Refine the bracket until we find a point satisfying the criteria
    insufProgress = 0
    Tpos = 1
    LOposRemoved = 0
    while not done and LSiter < maxLS:
        # Find High and Low Points in bracket
        LOpos = 0 if bracketFval[0]<=bracketFval[1] else 1
        f_LO = bracketFval[LOpos]
        HIpos = 1-LOpos

        #Compute new trial value
        if LS_interp <= 1 or not isLegal(bracketFval) or not isLegal(bracketGval):
            dprint('Bisecting')
            t = np.mean(bracket)
        elif LS_interp==2:
            dprint('Grad-Cubic Interpolation')
            t = polyinterp(bracket[0:2],bracketFval[0:2],[bracketGval[0].T@d,bracketGval[1].T@d],doPlot)
        else:
            # Mixed Case
            nonTpos = 1-Tpos
            if LOposRemoved == 0:
                oldLOval = bracket[nonTpos]
                oldLOFval = bracketFval[nonTpos]
                oldLOGval = bracketGval[nonTpos]
            t = mixedInterp(bracket,bracketFval,bracketGval,d,Tpos,oldLOval,oldLOFval,oldLOGval,dprint,doPlot)

        # Test that we are making sufficient progress
        maxbrac = max(bracket)
        minbrac = min(bracket)
        if minbrac != maxbrac and min(maxbrac-t,t-minbrac)/(maxbrac-minbrac) < 0.1:
            dprint('Interpolation close to boundary',end='')
            if insufProgress or t>=maxbrac or t<=minbrac:
                dprint(', Evaluating at 0.1 away from boundary')
                if abs(t-maxbrac) < abs(t-minbrac):
                    t = maxbrac-0.1*(maxbrac-minbrac)
                else:
                    t = minbrac+0.1*(maxbrac-minbrac)
                insufProgress = 0
            else:
                dprint('')
                insufProgress = 1
        else:
            insufProgress = 0

        # Evaluate new point
        if not saveHessianComp and useH:
            (f_new,g_new,H) = funObj(x+t*d,*args)[0:3]
        else:
            (f_new,g_new) = funObj(x+t*d,*args)[0:2]
        funEvals += 1
        gtd_new = g_new.T@d
        LSiter += 1

        armijo = f_new < (f+c1*t*gtd)
        if not armijo or f_new >= f_LO:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[HIpos] = t
            bracketFval[HIpos] = f_new
            bracketGval[HIpos] = g_new
            Tpos = HIpos
        else:
            if abs(gtd_new) <= -c2*gtd:
                # Wolfe conditions satisfied
                done = 1
            elif gtd_new*(bracket[HIpos]-bracket[LOpos]) >= 0:
                # old HI becomes new LO
                bracket[HIpos] = bracket[LOpos]
                bracketFval[HIpos] = bracketFval[LOpos]
                bracketGval[HIpos] = bracketGval[LOpos]
                if LS_interp == 3:
                    dprint('LO Pos is being removed!')
                    LOposRemoved = 1
                    oldLOval = bracket[LOpos]
                    oldLOFval = bracketFval[LOpos]
                    oldLOGval = bracketGval[LOpos]
            bracket[LOpos] = t
            bracketFval[LOpos] = f_new
            bracketGval[LOpos] = g_new
            Tpos = LOpos

        if not done and abs(bracket[0]-bracket[1])*nrmD < progTol:
            dprint('Line-search bracket has been reduced below progTol')
            break

    ##
    if LSiter == maxLS:
        dprint('Line Search Exceeded Maximum Line Search Iterations')

    LOpos = np.argmin(bracketFval)
    t = bracket[LOpos]
    f_new = bracketFval[LOpos]
    g_new = bracketGval[LOpos]

    # Evaluate Hessian at new point
    if useH and funEvals > 1 and saveHessianComp:
        (f_new,g_new,H) = funObj(x+t*d,*args)[0:3]
        funEvals += 1

    if useH:
        return (t,f_new,g_new,funEvals,H)
    else:
        return (t,f_new,g_new,funEvals)


def mixedExtrap(x,f,g,minStep,maxStep,dprint,doPlot):
    alpha_c = polyinterp(x,f,g,doPlot,minStep,maxStep)
    alpha_s = polyinterp(x,[f[0],None],g,doPlot,minStep,maxStep)
    if alpha_c > minStep and abs(alpha_c - x[1]) < abs(alpha_s - x[1]):
        dprint('Cubic Extrapolation')
        return alpha_c
    else:
        dprint('Secant Extrapolation')
        return alpha_s

def mixedInterp(x,f,g,d,Tpos,oldLOval,oldLOFval,oldLOGval,dprint,doPlot):
    # Mixed Case
    nonTpos = 1-Tpos
    gtdT = g[Tpos].T@d
    gtdNonT = g[nonTpos].T@d
    oldLOgtd = oldLOGval.T@d
    if f[Tpos] > oldLOFval:
        alpha_c = polyinterp([oldLOval,x[Tpos]],[oldLOFval,f[Tpos]],[oldLOgtd,gtdT],doPlot)
        alpha_q = polyinterp([oldLOval,x[Tpos]],[oldLOFval,f[Tpos]],[oldLOgtd,None],doPlot)
        if abs(alpha_c - oldLOval) < abs(alpha_q - oldLOval):
            dprint('Cubic Interpolation')
            return alpha_c
        else:
            dprint('Mixed Quad/Cubic Interpolation')
            return (alpha_q + alpha_c)/2
    elif gtdT*oldLOgtd < 0:
        alpha_c = polyinterp([oldLOval,x[Tpos]],[oldLOFval,f[Tpos]],[oldLOgtd,gtdT],doPlot)
        alpha_s = polyinterp([oldLOval,x[Tpos]],[oldLOFval,None],[oldLOgtd,gtdT],doPlot)
        if abs(alpha_c - x[Tpos]) >= abs(alpha_s - x[Tpos]):
            dprint('Cubic Interpolation')
            return alpha_c
        else:
            dprint('Quad Interpolation')
            return alpha_s
    elif abs(gtdT) <= abs(oldLOgtd):
        alpha_c = polyinterp([oldLOval,x[Tpos]],[oldLOFval,f[Tpos]],[oldLOgtd,gtdT],doPlot,min(x),max(x))
        alpha_s = polyinterp([oldLOval,x[Tpos]],[None,f[Tpos]],[oldLOgtd,gtdT],doPlot)
        if alpha_c > min(x) and alpha_c < max(x):
            dprint('Bounded Cubic Extrapolation')
            t = alpha_c
        else:
            dprint('Bounded Secant Extrapolation')
            t = alpha_s

        if x[Tpos] > oldLOval:
            t = min(x[Tpos] + 0.66*(x[nonTpos] - x[Tpos]),t)
        else:
            t = max(x[Tpos] + 0.66*(x[nonTpos] - x[Tpos]),t)

        return t
    else:
        return polyinterp([x[nonTpos],x[Tpos]],[f[nonTpos],f[Tpos]],[gtdNonT,gtdT],doPlot)

def polyinterp(x,f,g,doPlot=0,xminBound=None,xmaxBound=None):
# from the original Matlab (parameterization has been changed)
#
#   Minimum of interpolating polynomial based on function and derivative
#   values
#
#   It can also be used for extrapolation if {xmin,xmax} are outside
#   the domain of the points.
#
#   Input:
#       points(pointNum,[x f g])
#       doPlot: set to 1 to plot, default: 0
#       xmin: min value that brackets minimum (default: min of points)
#       xmax: max value that brackets maximum (default: max of points)
#
#   set f or g to sqrt(-1) if they are not known
#   the order of the polynomial is the number of known f and g values minus 1

# (sqrt(-1) replaced with None)

    nPoints = len(x)
    order = nPoints*2 - 1 - (f.count(None)+g.count(None))
    xmin = min(x)
    xmax = max(x)

    if xminBound is None: xminBound = xmin
    if xmaxBound is None: xmaxBound = xmax


    if nPoints == 2 and order==3 and doPlot==0:
        # Solution in this case (where x2 is the farthest point):
        #    d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
        #    d2 = sqrt(d1^2 - g1*g2);
        #    minPos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
        #    t_new = min(max(minPos,x1),x2);
        minPos = 0 if x[0]<x[1] else 1
        minVal = x[minPos]
        notMinPos = 1-minPos
        d1den = (x[minPos]-x[notMinPos])
        if d1den==0.0:
            return (xmaxBound+xminBound)/2
        d1 = g[minPos] + g[notMinPos] - 3*(f[minPos]-f[notMinPos])/d1den
        d2 = d1**2 - g[minPos]*g[notMinPos]
        if not np.any(d2 < 0):
            d2 = np.sqrt(d2)
            t = x[notMinPos]-(x[notMinPos]-x[minPos])*((g[notMinPos]+d2-d1)/(g[notMinPos]-g[minPos]+2*d2))
            return min(max(t,xminBound),xmaxBound)
        else:
            return (xmaxBound+xminBound)/2

    # Constraints Based on available Function Values
    A = np.zeros((0,order+1))
    b = np.zeros(0)
    for i in range(nPoints):
        if f[i] is not None:
            constraint = np.zeros((1,order+1))
            for j in range(order,-1,-1):
                constraint[0,order-j] = x[i]**j
            A = np.concatenate((A,constraint))
            b = np.concatenate((b,np.array([f[i]])))

    # Constraints based on available Derivatives
    for i in range(nPoints):
        if g[i] is not None:
            constraint = np.zeros((1,order+1))
            for j in range(order):
                constraint[0,j] = (order-j)*x[i]**(order-j-1)
            A = np.concatenate((A,constraint))
            b = np.concatenate((b,np.array([g[i]])))

    # Find interpolating polynomial
    params = np.linalg.solve(A,b)

    # Compute Critical Points
    dParams = np.zeros(order)
    for i,p in enumerate(params[:-1]):
        dParams[i] = p*(order-i)

    if np.any(np.isinf(dParams)):
        cp = np.concatenate((np.array([xminBound,xmaxBound],x)))
    else:
        cp = np.concatenate((np.array([xminBound,xmaxBound]),x,np.roots(dParams)))

    # Test Critical Points
    fmin = np.inf
    minPos = (xminBound+xmaxBound)/2 # Default to Bisection if no critical points valid
    for xCP in cp:
        if xCP is not None and np.isreal(xCP) and xCP >= xminBound and xCP <= xmaxBound:
            fCP = np.polyval(params,xCP)
            if np.isreal(fCP) and fCP < fmin:
                minPos = xCP
                fmin = fCP

    # Plot Situation
    if doPlot:
        import matplotlib.pyplot as plt

        plt.clf()
        plt.plot(x,f,'b*')
        for i in range(2):
            if g[i] is not None:
                m = g[i]
                b = f[i] - m*x[i]
                xs = np.array([x[i]-0.05,x[i]+0.05])
                plt.plot(xs,xs*m+b,'c.-')

        xs = np.linspace(min(xmin,xminBound)-.1,max(xmax,xmaxBound)+.1,100)
        plt.plot(xs,np.polyval(params,xs),'g+')
        if doPlot == 1:
            plt.pause(1)

    return minPos

def ArmijoBacktrack(x,t,d,f,fr,g,gtd,c1,LS_interp,LS_multi,progTol,dprint,doPlot,saveHessianComp,useH,funObj,*args):
# from original Matlab code:
# [t,x_new,f_new,g_new,funEvals,H] = ArmijoBacktrack(...
#    x,t,d,f,fr,g,gtd,c1,LS_interp,LS_multi,progTol,debug,doPlot,saveHessianComp,funObj,varargin)
#
# Backtracking linesearch to satisfy Armijo condition
#
# Inputs:
#   x: starting location
#   t: initial step size
#   d: descent direction
#   f: function value at starting location
#   fr: reference function value (usually funObj(x))
#   gtd: directional derivative at starting location
#   c1: sufficient decrease parameter
#   debug: display debugging information
#   LS_interp: type of interpolation
#   progTol: minimum allowable step length
#   doPlot: do a graphical display of interpolation
#   funObj: objective function
#   varargin: parameters of objective function
#
# Outputs:
#   t: step length
#   f_new: function value at x+t*d
#   g_new: gradient value at x+t*d
#   funEvals: number function evaluations performed by line search
#   H: Hessian at initial guess (only computed if requested)
#
# recet change: LS changed to LS_interp and LS_multi

    # Evaluate the Objective and Gradient at the Initial Step
    if useH:
        (f_new,g_new,H) = funObj(x+t*d,*args)[0:3]
    else:
        (f_new,g_new) = funObj(x+t*d,*args)[0:2]
    funEvals = 1

    while f_new > fr + c1*t*gtd or not isLegal(f_new):
        temp = t

        if LS_interp == 0 or not isLegal(f_new):
            # Ignore value of new point
            dprint('Fixed BT')
            t *= 0.5
        elif LS_interp == 1 or not isLegal(g_new):
            # Use function value at new point, but not its derivative
            if funEvals < 2 or LS_multi == 0 or not isLegal(f_prev):
                # Backtracking w/ quadratic interpolation based on two points
                dprint('Quad BT')
                t = polyinterp([0,t],[f,f_new],[gtd,None],doPlot,0,t)
            else:
                # Backtracking w/ cubic interpolation based on three points
                dprint('Cubic BT')
                t = polyinterp([0,t],[f,f_new],[None,None],doPlot,0,t)
        else:
            #Use function value and derivative at new point

            if funEvals < 2 or LS_multi == 0 or not isLegal(f_prev):
                # Backtracking w/ cubic interpolation w/ derivative
                dprint('Grad-Cubic BT')
                t = polyinterp([0,t],[f,f_new],[gtd,g_new.T@d],doPlot,0,t)
            elif not isLegal(g_prev):
                # Backtracking w/ quartic interpolation 3 points and derivative of two
                dprint('Grad-Quartic BT')
                t = polyinterp([0,t,t_prev],[f,f_new,f_prev],[gtd,g_new.T@d,None],doPlot,0,t)
            else:
                # Backtracking w/ quintic interpolation of 3 poitns and derivative of two
                dprint('Grad-Quintic BT')
                t = polyinterp([0,t,t_prev],[f,f_new,f_prev],[gtd,g_new.T@d,g_prev.T@d],doPlot,0,t)

        # Adjust if change in t is too small/large
        if t < temp*1e-3:
            dprint('Interpolated Value Too Small, Adjusting')
            t = temp*1e-3
        elif t > temp*0.6:
            dprint('Interpolated Value Too Large, Adjusting')
            t = temp*0.6

        # Store old point if doing three-point interpolation
        if LS_multi:
            f_prev = f_new
            t_prev = temp
            if LS_interp == 2: g_prev = g_new

        if not saveHessianComp and useH:
            (f_new,g_new,H) = funObj(x+t*d,*args)[0:3]
        else:
            (f_new,g_new) = funObj(x+t*d,*args)[0:2]
        funEvals += 1

        # Check whether step size has become too small
        if np.max(np.abs(t*d)) <= progTol:
            dprint('Backtracking Line Search Failed')
            t = 0
            f_new = f
            g_new = g
            break

    # Evaluate Hessian at new point
    if useH and funEvals>1 and saveHessianComp:
        (f_new,g_new,H) = funObj(x+t*d,*args)[0:3]
        funEvals += 1

    x_new = x+t*d
    if useH:
        return (t,x_new,f_new,g_new,funEvals,H)
    else:
        return (t,x_new,f_new,g_new,funEvals)


def conjGrad(A,b,optTol,maxIter,dprint,precFunc=None,precArgs=None,matrixVectFunc=None,matrixVectArgs=None,retnegCurv=False):
# original Matlab comments:
# [x,k,res,negCurv] =
# cg(A,b,optTol,maxIter,verbose,precFunc,precArgs,matrixVectFunc,matrixVect
# Args)
# Linear Conjugate Gradient, where optionally we use
# - preconditioner on vector v with precFunc(v,precArgs{:})
# - matrix multipled by vector with matrixVectFunc(v,matrixVectArgs{:})

    x = np.zeros(b.shape[0])
    r = -b

    # Apply preconditioner (if supplied)
    if precFunc is not None:
        y = precFunc(r,*precArgs)
    else:
        y = r

    ry = r.T@y
    p = -y
    k = 0

    res = np.linalg.norm(r)
    done = 0
    while res > optTol and k < maxIter and not done:
        # Compute Matrix-vector product
        if matrixVectFunc is not None:
            Ap = matrixVectFunc(p,*matrixVectArgs)
        else:
            Ap = A@p
        pAp = p.T@Ap

        # Check for negative Curvature
        if pAp <= 1e-16:
            dprint('Negative Curvature Detected!')

            if retnegCurv:
                if pAp < 0:
                    return (x,k,res,p)

            if k == 0:
                dprint('First-Iter, Proceeding...')
                done = 1
            else:
                dprint('Stopping')
                break

        # Conjugate Gradient
        alpha = ry/(pAp)
        x += alpha*p
        r += alpha*Ap
        
        # If supplied, apply preconditioner
        if precFunc is not None:
            y = precFunc(r,*precArgs)
        else:
            y = r
        
        ry_new = r.T@y
        beta = ry_new/ry
        p = -y + beta*p
        k += 1

        # Update variables
        ry = ry_new
        res = np.linalg.norm(r)

    if retnegCurv:
        return (x,k,res,None)
    else:
        return (x,k,res)

def taylorModel(d,f,g,H,T):
    p = d.shape[0]
    f = f.copy()
    g = g.copy()
    H = H.copy()
    fd3 = 0.
    gd2 = np.zeros(p)
    Hd = np.zeros((p,p))
    # cshelton: to replace with broadcasting code (currently dups Matlab code)
    for t1 in range(p):
        for t2 in range(p):
            for t3 in range(p):
                fd3 += T[t1,t2,t3]*d[t1]*d[t2]*d[t3]
                gd2[t3] += T[t1,t2,t3]*d[t1]*d[t2]
                Hd[t2,t3] += T[t1,t2,t3]*d[t1]
    f += g.T@d + 0.5*d.T@H@d + 1./6*fd3
    g += H@d + 0.5*gd2
    H += Hd
    if np.any(np.abs(d) > 1e5):
        # We want the optimizer to stop if the solution is unbounded
        g = np.zeros(p)
    return (f,g,H)
