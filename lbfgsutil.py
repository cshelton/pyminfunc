import numpy as np

# cshelton: much of this could probably be made faster with broadcasting
#  and tightening things up, but goal at the moment is to make it correct
#  (that is, the same as the Matlab code)

def lbfgsAdd(y,s,S,Y,YS,lbfgs_start,lbfgs_end,Hdiag,useMex):

    ys = y.T@s
    corrections = S.shape[1]
    if ys > 1e-10:
        if lbfgs_end < corrections:
            lbfgs_end += 1
            if lbfgs_start != 1:
                if lbfgs_start == corrections:
                    lbfgs_start = 1
                else:
                    lbfgs_start += 1
        else:
            lbfgs_start = min(2,corrections)
            lbfgs_end = 1

        if useMex: # same either way currently
            S[:,lbfgs_end-1] = s
            Y[:,lbfgs_end-1] = y
        else:
            S[:,lbfgs_end-1] = s
            Y[:,lbfgs_end-1] = y

        YS[lbfgs_end-1] = ys

        Hdiag = ys/(y.T@y)
        return (S,Y,YS,lbfgs_start,lbfgs_end,Hdiag,0)
    else:
        return (S,Y,YS,lbfgs_start,lbfgs_end,Hdiag,1)

def lbfgsProd(g,S,Y,YS,lbfgs_start,lbfgs_end,Hdiag):
# BFGS Search Direction
#
# This function returns the (L-BFGS) approximate inverse Hessian,
# multiplied by the negative gradient

    # Set up indexing
    (nVars,maxCorrections) = S.shape
    if lbfgs_start == 1:
        ind = list(range(lbfgs_end))
        nCor = lbfgs_end-lbfgs_start+1
    else:
        ind = list(range(lbfgs_start-1,maxCorrections)) + list(range(lbfgs_end))
        nCor = maxCorrections

    al = np.zeros(nCor)
    be = np.zeros(nCor)

    d = -g
    for i in reversed(ind):
        al[i] = (S[:,i].T@d)/YS[i]
        d = d-al[i]*Y[:,i]

    # Multiply by Initial Hessian
    d = Hdiag*d

    for i in ind:
        be[i] = (Y[:,i].T@d)/YS[i]
        d = d + S[:,i]*(al[i]-be[i])

    return d

def lbfgsUpdate(y,s,corrections,dprint,old_dirs,old_stps,Hdiag):
    ys = y.T@s
    if ys > 1e-10:
        numCorrections = old_dirs.shape[1]
        if numCorrections >= corrections:
            np.roll(old_dirs,-1,axis=1)
            np.roll(old_stps,-1,axis=1)
        old_dirs[:,numCorrections] = s
        old_stps[:,numCorrections] = y
        
        Hdiag = ys/(y.T@y)
    else:
        dprint('Skipping Update')
    return (old_dits,old_stps,Hdiag)

def lbfgs(g,s,y,Hdiag):
# comments from original Matlab code:
# BFGS Search Direction
#
# This function returns the (L-BFGS) approximate inverse Hessian,
# multiplied by the gradient
#
# If you pass in all previous directions/sizes, it will be the same as full BFGS
# If you truncate to the k most recent directions/sizes, it will be L-BFGS
#
# s - previous search directions (p by k)
# y - previous step sizes (p by k)
# g - gradient (p by 1)
# Hdiag - value of initial Hessian diagonal elements (scalar)

    (p,k) = s.shape

    # cshelton: unsure why this was a loop in the original code
    ro = 1/(y*s).sum(axis=0)

    q = np.zeros((p,k+1))
    r = np.zeros((p,k+1))
    al = np.zeros(k)
    be = np.zerso(k)

    q[:,k] = g
    for i in range(k-1,-1,-1):
        al[i] = ro[i]*s[:,i].T@q[:,i+1]
        q[:,i] = q[:,i+1] - al[i]*y[:,i]

    # Multiply by Initial Hessian
    r[:,1] = Hdiag*q[:,1]

    for i in range(k):
        be[i] = ro[i]*y[:,i].T@r[:,i]
        r[:,i+1] = r[:,i] + s[:,i]*(al[i]-be[i])

    return r[:,k+1]






