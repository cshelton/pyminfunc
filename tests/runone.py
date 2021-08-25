import sys
sys.path.insert(0,'../src')
from pyminfunc import minFunc
import numpy as np
from testfns import *

def runone(filename):
    with open(filename) as fid:
        line = fid.readline().rstrip()
        funname = line.split()[1]
        line = fid.readline().rstrip()
        x0 = np.fromstring(line,sep=' ')
        options = {}
        args = []
        procargs = False
        line = fid.readline().rstrip()
        while line:
            if line == '':
                procargs = true
            elif procargs:
                args.append(line)
            else:
                vals = line.split()
                fname = vals[0]
                fvalstr = vals[1]
                fval = vals[1]
                for type in (int,float):
                    try:
                        fval = type(fvalstr)
                    except ValueError:
                        continue
                options[fname] = fval
            line = fid.readline().rstrip()


    print('===START HERE')
    if len(args)==0:
        exestr = 'minFunc('+funname+',x0,options)'
    else:
        exestr = 'minFunc('+funname+',x0,options'+','.join(args)+')'

    if 'noutputs' not in options or options['noutputs'] == 3:
        x,f,exitflag = eval('minFunc('+funname+',x0,options)')
        output = None
    elif options['noutputs'] == 1:
        x = eval('minFunc('+funname+',x0,options)')
        f = None
        exitflag = None
        output = None
    elif options['noutputs'] == 2:
        x,f = eval('minFunc('+funname+',x0,options)')
        exitflag = None
        output = None
    elif options['noutputs'] == 4:
        x,f,exitflag,output = eval('minFunc('+funname+',x0,options)')

    print('x [ '+' '.join(['{:.10g}'.format(v) for v in x])+' ]')
    if f is not None: print(f'f {f:.10g}')
    if exitflag is not None: print('exitflag',exitflag)
    if output is not None:
        print('output.iterations',output.iterations)
        print('output.funcCount',output.funcCount)
        print(f'output.algorithm {output.algorithm:d}')
        print(f'output.firstorderopt {output.firstorderopt:.10g}')
        print('output.message',output.message)
        print('output.trace.fval',' '.join(['{:.10g}'.format(v) for v in output.trace.fval])+' ')
        print('output.trace.funcCount',' '.join(['{:d}'.format(v) for v in output.trace.funcCount])+' ')
        print('output.trace.optCond',' '.join(['{:.10g}'.format(v) for v in output.trace.optCond])+' ')
    print('===END HERE\n')

runone(sys.argv[1])
