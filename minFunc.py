import numpy as np
from numpy.linalg import norm
import scipy.linalg as la
from enum import IntEnum
from scipy.optimize import check_grad
from scipy.optimize import line_search
from collections import namedtuple
from util import *

class methods(IntEnum):
    SD = 0,
    CSD = 1,
    BB = 2,
    CG = 3,
    PCG = 4,
    LBFGS = 5,
    QNEWTON = 6,
    NEWTON0 = 7,
    NEWTON = 8,
    TENSOR = 9,
    SCG = 13,
    PNEWTON0 = 17,
    MNEWTON = 18

class minFuncoptions:
    @staticmethod
    def getoptbool(options,name,poslst,posval,negval):
        if name not in options:
            return negval
        elif options[name].upper() in poslst:
            return posval
        else: return negval

    @staticmethod
    def getopt(options,name,defval):
        if name not in options or options[name] == '':
            return defval
        else: return options[name]

    initdefaultParams = {
            'LS_init':0,
            'LS_type':1,
            'LS_interp':2,
            'LS_multi':0,
            'Fref':1,
            'Damped':0,
            'HessianIter':1,
            'c2':0.9,
            'cgSolve':0
            }

    methoddefaultParams = {
            methods.TENSOR:{},
            methods.NEWTON:{},
            methods.MNEWTON:{'method':methods.NEWTON,'HessianIter':5},
            methods.PNEWTON0:{'method':methods.PNEWTON0,'cgSolve':1},
            methods.NEWTON0:{},
            methods.QNEWTON:{'Damped':1},
            methods.LBFGS:{},
            methods.BB:{},
            methods.PCG:{'c2':0.2,'LS_init':2},
            methods.SCG:{'method':methods.CG,'c2':0.2,'LS_init':4},
            methods.CG:{'c2':0.2,'LS_init':2},
            methods.CSD:{'c2':0.2,'Fref':10,'LS_init':2},
            methods.SD:{'LS_init':2}
            }

    useexisting = object()

    optionparams = (('maxFunEvals','MAXFUNEVALS',1000),
                    ('maxIter','MAXITER',500),
                    ('optTol','OPTTOL',1e-5),
                    ('progTol','PROGTOL',1e-9),
                    ('corrections','CORRECTIONS',100),
                    ('corrections','CORR',useexisting),
                    ('c1','C1',1e-4),
                    ('c2','C2',useexisting),
                    ('LS_init','LS_INIT',useexisting),
                    ('cgSolve','CGSOLVE',useexisting),
                    ('qnUpdate','QNUPDATE',3),
                    ('cgUpdate','CGUPDATE',2),
                    ('initialHessType','INITIALHESSTYPE',1),
                    ('HessianModify','HESSIANMODIFY',0),
                    ('Fref','FREF',useexisting),
                    ('useComplex','USECOMPLEX',0),
                    ('numDiff','NUMDIFF',0),
                    ('LS_saveHessianComp','HS_SAVEHESSIANCOMP',1),
                    ('Damped','DAMPED',useexisting),
                    ('HvFunc','HVFUNC',None),
                    ('bbType','BBTYPE',0),
                    ('cycle','CYCLE',3),
                    ('HessianIter','HESSIANITER',useexisting),
                    ('outputFcn','OUTPUTFCN',None),
                    ('useMex','USEMEX',1),
                    ('useNegCurv','USENEGCURV',1),
                    ('precFunc','PRECFUNC',None),
                    ('LS_type','LS_TYPE',useexisting),
                    ('LS_interp','LS_INTERP',useexisting),
                    ('LS_multi','LS_MULTI',useexisting),
                    ('noutputs','NOUTPUTS',3),
                    )

    def __init__(self,options):
        uopt = {k.upper():v for k,v in options.items()} 
        self.verbose = self.getoptbool(uopt,'DISPLAY',(0,'OFF','NONE'),0,1)
        self.verboseI = self.getoptbool(uopt,'DISPLAY',(0,'OFF','NONE','FINAL'),0,1)
        self.debug = self.getoptbool(uopt,'DISPLAY',('FULL','EXCESSIVE'),1,0)
        self.doPlot = self.getoptbool(uopt,'DISPLAY',('EXCESSIVE',),1,0)
        self.checkGrad = self.getoptbool(uopt,'DERIVATIVECHECK',(1,'ON'),1,0)
        methodstr = self.getopt(uopt,'METHOD','LBFGS').upper()
        self.method = methods.__members__[methodstr]
        for field,value in minFuncoptions.initdefaultParams.items():
            setattr(self,field,value)
        for field,value in minFuncoptions.methoddefaultParams[self.method].items():
            setattr(self,field,value)
        for field,name,defvalue in minFuncoptions.optionparams:
            if defvalue is minFuncoptions.useexisting:
                setattr(self,field,self.getopt(uopt,name,getattr(self,field)))
            else:
                setattr(self,field,self.getopt(uopt,name,defvalue))

class traceT:
    def __init__(self,fval,funcCount,optCond):
        self.fval = fval
        self.funcCount = funcCount
        self.optCond = optCond

class outputT:
    def __init__(self,iterations,funcCount,algorithm,firstOrderopt,message,trace):
        self.iterations =iterations
        self.funcCount = funcCount
        self.algorithm = algorithm
        self.firstOrderopt = firstOrderopt
        self.message = message
        self.trace = trace

# (intentional or unavoidable) differences:
#  - derivative checking is done differently (and not on Hessians) -- could be fixed
#  - must specify the number of desired outputs (defaults to 3: x, f, exitflag)
#     by setting 'noutputs' to be the desired value (1,2,3, or 4) in options
#    python does not pass this information along like Matlab does
#  - useMex currently does nothing, but is passed around.  In the future, we could
#     add precompiled C code as an option in a similar fashion
def minFunc(funObj,x0,options,*args):
    o = minFuncoptions(options)
    p = x0.shape[0]
    d = np.zeros((p,))
    x = x0.astype(float)
    t = 1

    def dprint(*args,**kwargs):
        if o.debug:
            print(*args,**kwargs)

    def gatherret():
        if o.noutputs==1:
            return x
        if o.noutputs==2:
            return x,f
        if o.noutputs==3:
            return x,f,exitflag
        if o.noutputs==4:
            return x,f,exitflag,output

    funEvalMultiplier = 1
    numDiffType = 3 if o.useComplex else o.numDiff
    if o.numDiff and o.method != methods.TENSOR:
        args = [numDiffType,funObj] + args
        if o.method != methods.NEWTON:
            if o.useComplex:
                dprint('Using complex differentials for gradient computation')
            else:
                dprint('Using finite differences for gradient computation')
            funObj = autoGrad
        else:
            if o.useComplex:
                dprint('Using complex differentials for Hessian computation')
            else:
                dprint('Using finite differences for Hessian compuation')
            funObj = autoHess

        if o.method == methods.NEWTON0 and o.useComplex:
            dprint('Turning off the use of complex differentials for Hessian-vector products')
            o.useComplex = 0

        if o.useComplex:
            funEvalMultiplier = p
        elif o.numDiff == 2:
            funEvalMultiplier = 2*p
        else:
            funEvalMultipler = p+1

    if o.method < methods.NEWTON:
        f,g = funObj(x,*args)
        computeHessian = 0
    else:
        f,g,H = funObj(x,*args)
        computeHessian = 1
    funEvals = 1

    if o.checkGrad:
        if o.numDiff:
            print('Can not do derivative checking when numDiff is 1')
        else:
            # cshelton 7/7/21: guess at how to implement corresponding "derivativeCheck"
            # calls from Matlab code
            # scipy seems to only use forward differences, so this isn't completely the same
            if check_grad(lambda x : funObj(x,*args)[0], lambda x : funObj(x,*args)[1])>1e-6:
                print('derivative check does not match')
            # to do? 7/7/21: implement Hessian check (if computeHessian is true)
    
    if o.verboseI:
        print('{:>10} {:>10} {:>15} {:>15} {:15}'.format('Iteration','FunEvals','Step Length','Function Val','Opt Cond'))

    optCond = np.max(np.abs(g))

    if o.noutputs>3:
        trace = traceT(f,funEvals,optCond)

    if optCond <= o.optTol:
        exitflag = 1
        msg = 'Optimality Condition below topTol'
        if o.verbose: print(msg)
        if o.noutputs>3:
            output = outputT(0,1,o.method,np.max(np.abs(g)),msg,trace)
        return gatherret()

    if o.outputFcn is not None:
        stop = o.outputFcn(x,'init',0,funEvals,f,None,None,g,None,np.max(np.abs(g)),*args)
        if stop:
            exitflag = 1
            msg = 'Stopped by output function'
            if o.verbose: print(msg)
            if o.noutputs>3:
                output = outputT(0,1,o.method,np.max(np.abs(g)),trace)
            return gatherret()

    for i in range(1,o.maxIter+1):
        # to do? replace with match when in python 3.10?
        #      or replace with dictionary dispatch?
        if o.method==methods.SD: # Steepest Descent
            d = -g
        elif o.method==methods.CSD: # Cyclic Steepest Descent
            if i % o.cycle == 1:
                alpha = 1
                o.LS_init = 2
                o.LS_type = 1
            elif i % o.cycle == 2 % o.cycle:
                alpha = t
                o.LS_init = 0
                o.LS_type = 0
            d = -alpha*g
        elif o.method==methods.BB: # Steepest Descent with Barzilai and Borwein Step Length
            if i==1:
                d = -g
            else:
                y = g-g_old
                s = t*d
                if o.bbType == 0:
                    alpha = (s.T@y)/(y.T@y)
                    if alpha <= 1e-10 or alpha > 1e10:
                        alpha = 1
                elif o.bbType == 1:
                    alpha = (s.T@s)/(s.T@y)
                    if alpha <= 1e-10 or alpha > 1e10:
                        alpha = 1
                elif o.bbType == 2: # Conic Interpolation ('Modified BB')
                    ss = (s.T@s).item()
                    alpha = ss/(s.T@y)
                    if alpha <= 1e-10 or alpha > 1e10:
                        alpha = 1
                    alphaConic = ss/(6*(myF_old - f) + 4*g.T@s + 2*g_old.T@s)
                    if alphaConic > 0.001*alpha and alphaConic < 1000*alpha:
                        alpha = alphaConic
                elif o.bbType == 3: # Gradient Method with retards (bb type 1, random selection of previous step)
                    alpha = (s.T@s)/(s.T@y)
                    if alpha < 1e-10 or alpha > 1e10:
                        alpha = 1
                    if i<=2:
                        v = np.zeros(5)
                    v[(i-2)%5] = alpha
                    alpha = v[np.random.randint(min(i-2,5))]
                d = -alpha*g
            g_old = g
            myF_old = f
        elif o.method==methods.CG: # Non-linear Conjugate Gradient
            if i==1:
                d = -g
            else:
                gotgo = g_old.T@g_old
                if o.cgUpdate == 0:
                    # Fletcher-Reeves
                    beta = g.T@g / gotgo
                elif cgUpdate == 1:
                    # Polak-Ribiere
                    beta = (g.T@(g-g_old)) / gotgo
                elif cgUpdate == 2:
                    # Hestenes-Stiefel
                    beta = (g.T@(g-g_old)) / ((g-g_gold).T@d)
                else:
                    # Gilbert-Nocedal
                    beta_FR = (g.T@(g-g_old)) / gotgo
                    beta_PR = (g.T@g - g.T@g_old) / gotgo
                    beta = np.max(-beta_FT,np.min(beta_PR,beta_FR))
                d = -g + beta*d

                # Restart if not a direction of sufficient descent
                if g.T@d > -o.progTol:
                    dprint('Restarting CG')
                    beta = 0
                    d = -g
            g_old = g
        elif o.method == methods.PCG: # Preconditioned Non-Linear Conjugate Gradient
            if o.precFunc is None:
                if i==1:
                    S = np.zeros((p,o.corrections))
                    Y = np.zeros((p,o.corrections))
                    YS = np.zeros((o.corrections,))
                    lbfgs_start = 1
                    lbfgs_end = 0
                    Hdiag = 1
                    s = -g
                else:
                    (S,Y,YS,lbfgs_start,lbfgs_end,Hdiag,skipped) = lbfgsAdd(g-g_old,t*d,S,Y,YS,lbfgs_start,lbfgs_end,Hdiag,o.useMex)
                    if skipped: dprint('Skipped L-BFGS updated')
                    if o.useMex:
                        precondFunc = lbfgsProd # same either way, currently
                    else:
                        precondFunc = lbfgsProd # same either way, currently
            else:
                s = o.precFunc(-g,x,*args)

            if i==1:
                d = s
            else:
                if o.cgUpdate == 0:
                    # Preconditioned Fletcher-Reeves
                    beta = (g.T@s)/(g_old.T@s_old)
                elif o.cgUpdate < 3:
                    # Preconditioned Polak-Ribiere
                    beta = (g.T@(s-s_old))/(g_old.T@s_old)
                else:
                    # Preconditioned Gilbert-Nocedal
                    beta_FR = (g.T@s)/(g_old.T@s_old)
                    beta_PR = (g.T@(s-s_old))/(g_old.T@s_old)
                    beta = np.max(-beta_FR,np.min(beta_PR,beta_FR))
                d = s + beta*d
                if g.T@d > -o.progTol:
                    dprint('Restarting CG')
                    beta = 0
                    d = s
            g_gold = g
            s_old = s
        elif o.method==methods.LBFGS: # L-BFGS
            if o.Damped:
                if i==1:
                    d = -g # Initially use steepest descent direction
                    old_dirs = np.zeros((g.shape[0],0))
                    old_stps = np.zeros((d.shape[0],0))
                    Hdiag = 1
                else:
                    old_dirs,old_stps,Hdiag = dampedUpdate(g-g_old,t*d,o.corrections,o.debug,old_dirs,old_stps,Hdiag)
                    if o.useMex:
                        d = lbfgs(-g,old_dirs,old_stps,Hdiag) # currently same either way
                    else:
                        d = lbfgs(-g,old_dirs,old_stps,Hdiag) # currently same either way
            else:
                if i==1:
                    d = -g # Initially use steepest descent direction
                    S = np.zeros((p,o.corrections))
                    Y = np.zeros((p,o.corrections))
                    YS = np.zeros((o.corrections,))
                    lbfgs_start = 1
                    lbfgs_end = 0
                    Hdiag = 1
                else:
                    S,Y,Ys,lbfgs_start,lbfgs_end,Hdiag,skipped = lbfgsAdd(g-g_old,t*d,S,Y,YS,lbfgs_start,lbfgs_end,Hdiag,o.useMex)
                    if skipped: dprint('Skipped L-BFGS updated')
                    if o.useMex:
                        d = lbfgsProc(g,S,Y,Ys,lbfgs_start,lbfgs_end,Hdiag) # currently the same either way
                    else:
                        d = lbfgsProc(g,S,Y,Ys,lbfgs_start,lbfgs_end,Hdiag) # currently the same either way
            g_old = g
        elif o.method==methods.QNEWTON:
            if i==1:
                d = -g
            else:
                y = g-g_old
                s = t*d
                if i==2:
                    # Make initial Hessian approximation
                    if o.initialHessType == 0:
                        # Identity
                        if o.qnUpdate <= 1:
                            R = np.eye(g.shape[0])
                        else:
                            H = np.eye(g.shape[0])
                    else:
                        dprint('Scaling Initial Hessian Approximation')
                        if qnUpdate <= 1:
                            # Use Cholesky of Hessian approximation
                            R = np.sqrt((y.T@y)/(y.T@s))*np.eye(g.shape[0])
                        else:
                            H = np.eye(g.shape[0])*(y.T@s)/(y.T@y)
                if o.qnUpdate == 0: # Use BFGS updates
                    Bs = R.T@(R*s)
                    if o.Damped:
                        eta = 0.02
                        if y.T@s < eta*s.T@Bs:
                            dprint('Damped Update')
                            theta = np.min(np.max(0,((1-eta)*s.T@Bs)/(s.T@Bs - y.T@s)),1)
                            y = theta*y + (1-theta)*Bs
                        R = cholupdate(cholupdate(R,y/np.sqrt(y.T@s)),Bs/np.sqrt(s.T@Bs),'-')
                    else:
                        if y.T@s > 1e-10:
                            R = cholupdate(cholupdate(R,y/np.sqrt(y.T@s)),Bs/np.sqrt(s.T@Bs),'-')
                        else:
                            dprint('Skipping Update')
                elif o.qnUpdate == 1: # Perform SR1 Update if it maintains positive-definiteness
                    Bs = R.T@(R@s)
                    ymBs = y-Bs
                    if np.abs(s.T@ymBs) >= norm(s)*norm(ymBs)*1e-8 \
                            and (s-((mldivide(R,mldivide(R.T,y))))).T@y > 1e-10:
                        R = cholupdate(R,-ymBs/np.sqrt(ymBs.T@s),'-')
                    else:
                        dprint('SR1 not positive-definite, doing BFGS Update')
                        if o.Damped:
                            eta = 0.02
                            if y.T@s < eta*s.T@Bs:
                                dprint('Damped Update')
                                theta = min(max(0,((1-eta)*s.T@Bs)/(s.T@Bs - y.T@s)),1)
                                y = theta*y + (1-theta)*Bs
                            R = cholupdate(cholupdate(R,y/np.sqrt(y.T@s)),Bs/np.sqrt(s.T@Bs),'-')
                        else:
                            if y.T@s > 1e-10:
                                R = cholupdate(cholupdate(R,y/np.sqrt(y.T@s)),Bs/np.sqrt(s.T@Bs),'-')
                            else:
                                dprint('Skipping Update')
                elif o.qnUpdate == 2: # Use Hoshino update
                    v = np.sqrt(y.T@H@y)*(s/(s.T@y) - (H@y)/(y.T@H@y))
                    phi = 1./(1. + (y.T@H@y)/(s.T@y))
                    H = H + np.outer(s,s)/(s.T@y) - np.outer(H@y,y)@H/(y.T@H@y) + phi*np.outer(v,v)
                elif o.qnUpdate == 3: # Self-Scaling BFGS update
                    ys = y.T@s
                    Hy = H@y
                    yHy = y.T@Hy
                    gamma = ys/yHy
                    v = np.sqrt(yHy)*(s/ys - Hy/yHy)
                    H = gamma*(H - np.outer(Hy,Hy)/yHy + np.outer(v,v)) + np.outer(s,s)/ys
                elif o.qnUpdate == 4: # Oren's Self-Scaling Variable Metric update
                    # Oren's method
                    if (s.T@y)/(y.T@H@y) > 1:
                        phi = 1 # BFGS
                        omega = 0
                    elif (s.T@mldivide(H,s))/(s.T@y) < 1:
                        phi = 0 # DFP
                        omega = 1
                    else:
                        phi = (s.T@y)*(y.T@H@y - s.T@y)/((s.T@mldivide(H,s))*(y.T@H@y)-(s.T@y)**2)
                        omega = phi
                    gamma = (1-omega)*(s.T@y)/(y.T@H@y) + omega*(s.T@(mldivide(H,s)))/(s.T@y)
                    v = np.sqrt(y.T@H@y)*(s/(s.T@y) - (H@y)/(y.T@H@y))
                    H = gamma*(H - (np.outer(H@y,y)@H)/(y.T@H@y) + phi*np.outer(v,v)) + np.outer(s,s)/(s.T@y)
                elif o.qnUpdate == 5:
                    theta = 1
                    phi = 0
                    psi = 1
                    omega = 0
                    t1 = np.outer(s,(theta*s + phi*h.T@y))
                    t2 = (theta*s + phi*H.T@y).T@y
                    t3 = np.outer(H@y,psi*s + omega*H.T@y)
                    t4 = (psi*s + omega*H.T@y).T*y
                    H = H + t1/t2 - t3/t4

                if o.qnUpdate <= 1:
                    d = -mldivide(R,mldivide(R.T,g))
                else:
                    d = -H@g
            g_old = g
        elif o.method==methods.NEWTON0: # Hessian-Free Newton
            cgMaxIter = np.min(p,o.maxFunEval-funEvals)
            cgForce = np.min(0.5,np.sqrt(norm(g)))*norm(g)

            precondFunc = None
            precondArgs = ()
            if o.cgSolve == 1:
                if o.precFunc is None: # Apply L-BFGS preconditioner
                    if i==1:
                        S = np.zeros((p,o.corrections))
                        Y = np.zeros((p,o.corrections))
                        YS = np.zeros((o.corrections,))
                        lbfgs_start = 1
                        lbfgs_end = 0
                        Hdiag = 1
                    else:
                        S,Y,YS,lbfgs_start,lbfgs_end,Hdiag,skipped = lbfgsAdd(g-g_old,t*d,S,Y,YS,lbfgs_start,lbfgs_end,Hdiag,o.useMex)
                        if skipped: dprint('Skipped L-BFGS updated')
                        if o.useMex:
                            precondFunc = lbfgsProd # currently the same either way
                        else:
                            precondFunc = lbfgsProd # currently the same either way
                        precondArgs = (S,Y,Ys,lbfgs_start,lbfgs_end,Hdiag)
                    g_old = g
                else:
                    precondFunc = o.precFunc
                    precondArgs = (x,*args)
            #Solve Newton system using cg and hessian-vector products
            if o.HvFunc is None:
                # No user-supplied Hessian-vector function,
                # use automatic differentiation
                HvFun = autoHv
                HvArgs = (x,g,o.useComplex,funObj,*args)
            else:
                # Use user-supplied Hessian-vector function
                HvFun = o.HvFunc
                HvArgs = (x,*args)

            if o.useNegCurv:
                d,cgIter,cgRes,negCurv = conjGrad(None,-g,cgForce,cgMaxIter,o.debug,precondFunc,precondArgs,HvFun,HvArgs,True)
            else:
                d,cgIter,cgRes = conjGrad(None,-g,cgForce,cgMaxIter,o.debug,precondFunc,precondArgs,HvFun,HvArgs,False)

            funEvals += cgIter
            dprint(f'newtonCG stopped on iteration {cgIter} w/ residual {cgRes:.5e}')

            if o.useNegCurv:
                if negCurv is not None:
                    dprint('Using negative curvature direction')
                    d = negCurv/norm(negCurv)
                    d = d/np.sum(np.abs(g))

        elif o.method==methods.NEWTON:
            if o.cgSolve==0:
                if o.HessianModify==0:
                    # Attempt to perform Cholesky factorization of the Hessian
                    try:
                        R = la.cholesky(H)
                        d = -mldivide(R,mldivide(R.T,g))
                    except la.LinAlgError:
                        # otherwise, adjust the Hessian to be positive definite base on the
                        # minimum eigenvalue, and solve with QR
                        # (expensive, we don't want to do this very much)
                        dprint('Adjusting Hessian')
                        H = H + np.eye(g.shape[0])*max(0,1e-12 - np.min(np.real(np.eigh(H)[0])))
                        d = -mldivide(H,g)
                elif o.HessianModify==1:
                    # Modified Incomplete Cholesky
                    R = mcholinc(H,o.debug)
                    d = -mldivide(R,mldivide(R.T,g))
                elif o.HessianModify==2:
                    # Modified Generalized Cholesky
                    if o.useMex:
                        L,D,perm = mchol(H) # currently the same either way
                    else:
                        L,D,perm = mchol(H) # currently the same either way
                    d[perm] = mldivide(-L.T,(1./D * mldivide(L,g[perm])))
                elif o.HessianModify==3:
                    #Modified Spectral Decomposition
                    D,V = la.eig((H+H.T)/2.)
                    D = np.max(np.abs(D),np.max(np.max(np.abs(D)),1)*1e-12)
                    d = -V@((V.T@g)/D)
                elif o.HessianModify==4:
                    # Modified Symmetric Indefinite Factorization
                    L,D,perm = la.ldl(H)
                    # cshelton: pretty sure this is the same as the matlab code
                    for i in range(p):
                        if i<p-1 and D[i,i+1]!=0.0: # cshelton: first row of 2x2 block
                           block_a = D[i,i]
                           block_b = D[i+1,i]
                           block_d = D[i+1,i+1]
                           lam = (block_a+block_d)/2 - np.sqrt(4*block_b**2 + (block_a-block_d)**2)/2
                           D[i:i+1,i:i+1] = D[i:i+1,i:i+1]+np.eye(2)*(lam+1e-12)
                        elif (i==0 or D[i-1,i]==0) and D[i,i] < 1e-12:
                                    # not a row of a 2x2 block and diagonal is too smalle
                                D[i,i] = 1e-12
                    D[perm,:] = mldivide(-L.T,mldivide(D,mldivide(L,g[perm])))
                else:
                    # Take Newton step if Hessian is pd
                    # otherwise take a step with negative curvature
                    try:
                        R = la.cholesky(H)
                        d = -mldivide(R,mldivide(R.T,g))
                    except la.LinAlgError:
                        dprint('Taking Direction of Negative Curvature')
                        D,V = np.eigh(H)
                        U = V[:,0]
                        d = -np.sign(u^T@g)*u
            else:
                # Solve with Conjugate Gradient
                cgMaxIter = p
                cgForce = np.min(0.5,np.sqrt(norm(g)))*norm(g)
                # Select Preconditioner
                if o.cgSolve == 1:
                    # No preconditioner
                    precondFunc = None
                    precondArgs = None
                elif o.cgSolve == 2:
                    # Diagonal preconditioner
                    precDiag = np.diag(H)
                    precDiag[precDiag<1e-12] = 1e-12 - np.min(precDiag)
                    precondFunct = precondDiag
                    precondArgs = (1.0/precDiag,)
                elif o.cgSolve == 3:
                    # L-BFGS preconditioner
                    if i == 1:
                        old_dirs = np.zeros((g.shape[0],0))
                        old_stps = np.zeros((g.shape[0],0))
                        Hdiag = 1
                    else:
                        old_dirs,old_stps,Hdiag = lbfgsUpdate(g-g_old,t*d,o.corrections,o.debug,old_dirs,old_stps,Hdiag)
                    g_old = g
                    if o.useMex:
                        precondFunc = lbfgs # currently same either way
                    else:
                        precondFunc = lbfgs # currently same either way
                    precondArgs = (old_dirs,old_stps,Hdiag)
                elif o.cgSolve > 0:
                    # Symmetric Successive Overelaxation Preconditioner
                    omega = o.cgSolve
                    D = np.diag(H)
                    D[D<1e-12] = 1e-12 - np.min(D)
                    precDiag = (omega/(2-omega))/D
                    precTriu = np.diag(D/omega) + np.triu(H,1)
                    precondFunc = precondTriuDiag
                    precondArgs = (preTriu,1.0/precDiag)
                else:
                    # Incomplete Cholesky Preconditioner
                    R = ichol(H,droptol=-o.cgSolve,rdiag=1)
                    if np.min(np.diag(R)) < 1e-12:
                        R = ichol(H+np.eye(1e-12 - np.min(np.diag(R))),droptol=-o.cgSolve,rdiag=1)
                    precondFunc = precondTriu
                    precondArgs = (R,)

                # Run cg with the appropriate preconditioner
                if HvFunc is None:
                    # No user-supplied Hessian-vector function
                    d,cgIter,cgRes = confGrad(H,-g,cgForce,cgMaxIter,o.debug,precondFunc,precondArgs)
                else:
                    # Use user-supplied Hessian-vector function
                    d,cgIter,cgRes = conjGrad(H,-g,cgForce,cgMaxIter,o.deubg,precondFunc,precondArgs,HvFunc,(x,*args))
                dprint(f'CG stopped after {cgIter} iterations w/ residual {cgRes:.5e}')
        elif o.method == methods.TENSOR:
            if o.numDiff:
                T = autoTensor(x,numDiffType,funObj,*args)[3]
            else:
                T = funObj(x,*args)[3]
            d = minFunc(taylorModel,np.zeros((p,1)),
                    {'Method':'newton','Display':'none','progTol':o.progTol,'optTol':o.optTol},
                    f,g,H,T)

            if np.any(np.abs(d) > 1e5) or np.all(np.abs(d) < 1e-5) or g.T@d > -o.progTol:
                dprint('Using 2nd-Order Step')
                D,V = np.eigh((H+H.T)/2.)
                D = np.max(np.abs(D),np.max(np.max(np.abs(D)),1)*1e-12)
                d = -V@((V.T@g)/D)
            else:
                dprint('Using 3rd-Order Step')

        if not isLegal(d):
            print('Step direction is illegal!')
            return

        # ************** COMPUTE STEP LENGTH **************

        # Directional Derivative
        gtd = g.T@d

        # Check that progress can be made along direction
        if gtd > -o.progTol:
            exitflag = 2
            msg = 'Directional Derivative below progTol'
            break

        # Select Initial Guess
        if i == 1:
            if o.method < methods.NEWTON0:
                t = min(1.,1./np.sum(np.abs(g)))
            else:
                t = 1
        else:
            if o.LS_init == 0:
                # Newton step
                t = 1
            elif o.LS_init == 1:
                # Close to previous step length
                t = t*min(2,(gtd_old)/gtd)
            elif o.LS_init == 2:
                # Quadratic Initialization based on {f,g} and previous f
                t = min(1,2*(f-f_old)/gtd)
            elif o.LS_init == 3:
                # Double previous step length
                t = min(1,t*2)
            elif o.LS_init == 4:
                # Scaled step length if possible
                if HVFunc is None:
                    # No user-supplied Hessian-vector function
                    # use automatic differentiation
                    dHd = d.T@autoHv(d,g,g,0,funObj,*args)
                else:
                    # Use user-supplid Hessian-vector function
                    dHd = d.T@HvFunc(d,x,*args)

                funEvals += 1
                if dHd > 0:
                    t = -gtd/dHd
                else:
                    t = np.min(1,2*(f-f_old)/gtd)

            if t <= 0:
                t = 1

        f_old = f
        gtd_old = gtd

        # Compute reference fr is using non-monotone objective
        if o.Fref == 1:
            fr = f
        else:
            if i == 1:
                old_fvals = np.fill((o.Fref,1)-np.inf)

            if i <= o.Fref:
                old_fvals[i] = f
            else:
                old_fvals = np.concatenate((old_fvals[0:-1],np.array(f)))
            fr = np.max(old_fvals)

        computeHessian = 0
        if o.method >= methods.NEWTON:
            if o.HessianIter == 1:
                computeHessian = 1
            elif i>1 and (i-1) % o.HessianIter == 0:
                computeHessian = 1

        # Line Search
        f_old = f
        if o.LS_type == 0: # Use Armijo Bactracking
            # Perform Backtracking line search
            if computeHessian:
                t,x,f,g,LSfunEvals,H = ArmijoBacktrack(x,t,d,f,fr,g,gtd,o.c1,o.LS_interp,o.LS_multi,o.progTol,dprint,o.doPlot,o.LS_saveHessianComp,True,funObj,*args)
            else:
                t,x,f,g,LSfunEvals = ArmijoBacktrack(x,t,d,f,fr,g,gtd,o.c1,o.LS_interp,o.LS_multi,o.progTol,dprint,o.doPlot,1,False,funObj,*args)
            funEvals += LSfunEvals
        elif o.LS_type == 1: # Find Point satisfying Wolfe conditions
            if computeHessian:
                t,f,g,LSfunEvals,H = WolfeLineSearch(x,t,d,f,g,gtd,o.c1,o.c2,o.LS_interp,o.LS_multi,25,o.progTol,dprint,o.doPlot,o.LS_saveHessianComp,True,funObj,*args)
            else:
                t,f,g,LSfunEvals = WolfeLineSearch(x,t,d,f,g,gtd,o.c1,o.c2,o.LS_interp,o.LS_multi,25,o.progTol,dprint,o.doPlot,True,False,funObj,*args)
            funEvals += LSfunEvals
            x += t*d
        else: # Use toolbox line search
            # originally used Matlab optim toolbox line search
            # cannot find documentation for this, currently skipping
            raise NotImplementedError

        optCond = np.max(np.abs(g))
        if o.verboseI:
            print('{:10d} {:10d} {:15.5e} {:15.5e} {:15.5e}'.format(i,funEvals*funEvalMultiplier,t,f,optCond))

        if o.noutputs>3: # update trace
            trace.fval = np.append(trace.fval,f)
            trace.funcCount = np.append(trace.funcCount,funEvals)
            trace.optCond= np.append(trace.optCond,optCond)

        if o.outputFcn is not None: # output function
            if o.outputFcn(x,'iter',i,funEvals,f,t,gtd,g,d,optCond,*args):
                exitflag = -1
                msg = 'Stopped by output function'
                break

        if optCond <= o.optTol: # check optimality condition
            exitflag = 1
            msg = 'Optimality Condition below optTol'
            break

        # Check for lack of progress

        if np.max(np.abs(t*d)) <= o.progTol:
            exitflag = 2
            msg = 'Step Size below progTol'
            break

        if np.abs(f-f_old) < o.progTol:
            exitflag = 2
            msg = 'Function Value changing by less than progTol'
            break

        # check for going over iteration/evalutation limit

        if funEvals*funEvalMultiplier >= o.maxFunEvals:
            exitflag = 0
            msg = 'Reached Maximum Number of Function Evaluations'
            break

        if i == o.maxIter:
            exitflag = 0
            msg = 'Reached Maximum Number of Iterations'
            break


    if o.verbose: print(msg)
    if o.noutputs > 3:
        output = outputT(i,funEvals*funEvalMultiplier,method,np.max(np.abs(g)),msg,trace)

    if o.outputFcn is not None:
        o.outputFcn(x,'done',i,funEvals,f,t,gtd,g,d,np.max(np.abs(g)),*args)

    return gatherret()
