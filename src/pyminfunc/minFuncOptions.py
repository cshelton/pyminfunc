# Version 0.9 
# Aug 21, 2021
# by Christian Shelton
# based on code by Mark Schmidt

from enum import IntEnum

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
            methods.PNEWTON0:{'method':methods.NEWTON0,'cgSolve':1},
            methods.NEWTON0:{},
            methods.QNEWTON:{'Damped':1},
            methods.LBFGS:{},
            methods.BB:{'LS_type':0,'Fref':20},
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
        if options is None:
            options = {}
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
