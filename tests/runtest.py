#!/usr/bin/python
import sys
import os
import glob
import subprocess
from runpython import runpython
from runmatlab import runmatlab, runmatlabcont

def checkoutput(out1,out2):
    if out1==out2:
        return (True,None)

    ls1 = out1.split('\n')
    ls2 = out2.split('\n')
    if len(ls1) != len(ls2):
        return (False,"diff # of lines")
    numdiff = 0
    for l1,l2 in zip(ls1,ls2):
        if l1 != l2:
            ws1 = l1.split()
            ws2 = l2.split()
            if len(ws1) != len(ws2):
                return (False,None)
            for w1,w2 in zip(ws1,ws2):
                try:
                    f1 = float(w1)
                    f2 = float(w2)
                    rdiff = abs(f1-f2)/max(abs(f1),abs(f2),1e-6)
                    numdiff = max(numdiff,rdiff)
                except ValueError:
                    if w1 != w2:
                        return (False,w1+" "+w2)
    return (True,numdiff)

def runset(fnames):

    mlproc = None

    for testfn in fnames:
        fnstem = os.path.splitext(os.path.basename(testfn))[0]

        #pyout = subprocess.check_output(['./runpython.py',testfn]).decode('utf-8')
        #mlout = subprocess.check_output(['./runmatlab.py',testfn]).decode('utf-8')
        pyout = runpython(testfn)
        mlout,mlproc = runmatlabcont(testfn,mlproc)

        success,lvl = checkoutput(pyout,mlout)
        if success:
            if lvl is None:
                print(fnstem,'PASS')
            else:
                print(fnstem,f'PASS (@ {lvl:.5g})')
        else:
            print(fnstem,f'FAIL ({lvl})')
        with open('results/'+fnstem+'.pyout','w') as pyoutf:
            pyoutf.write(pyout)
        with open('results/'+fnstem+'.mlout','w') as mloutf:
            mloutf.write(mlout)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        runset(sorted(glob.glob('specs/*.txt')))
    else:
        runset(sys.argv[1:])
