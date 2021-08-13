import sys
sys.path.insert(0,'../')
from minFunc import minFunc

def runone(filename):
    with open(filename) as fid:
        line = fid.readline()
        funname = line
        line = fid.readline()
        x0 = np.fromstring(line,sep=' ')
        options = {}
        args = []
        procargs = False
        line = fid.readline()
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
            line = fid.readline()

    print(options)

    print('===START HERE')
    if len(procargs)==0:
        eval('(x,f,exitflag,output) = minFunc('+funname+',x0,options)')
    else:
        eval('(x,f,exitflag,output) = minFunc('+funname+',x0,options'+','.join(args)+')')

    print('x',x)
    print('f',f)
    print('exitflag',exitflag)
    print('output.iterations',output.iterations)
    print('output.funcCount ',output.funcCount)
    print('output.algorithm ',output.algorithm)
    print('output.firstorderopt ',output.firstorderopt)
    print('output.message ',output.message)
    print('output.trace.fval ',' '.join(output.trace.fval))
    print('output.trace.funcCount ',' '.join(output.trace.funcCount))
    print('output.trace.optCond ',' '.join(output.trace.optCond))
    print('===END HERE\n')

runone(argv[0])
