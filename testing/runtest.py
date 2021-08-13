#!/usr/bin/python
import sys
import subprocess

for testfn in sys.argv[1:]:

    pyout = subprocess.check_output(['./runpython.py',testfn]).decode('utf-8')
    mlout = subprocess.check_output(['./runmatlab.py',testfn]).decode('utf-8')

    if pyout==mlout:
        print(testfn,'PASS')
    else:
        with open(testfn+'.pyout','w') as pyoutf:
            pyoutf.write(pyout)
        with open(testfn+'.mlout','w') as mloutf:
            mloutf.write(mlout)
        print(testfn,'FAIL')

