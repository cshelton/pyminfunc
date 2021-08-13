#!/usr/bin/python

import subprocess
import sys

testfile = sys.argv[1]
output = subprocess.check_output(['matlab','-nodesktop','-nojvm','-nosplash','-r','try, runone(\''+testfile+'\'); end; quit']).decode('utf-8')

foundstart = False


for l in output.split('\n'):
    if foundstart:
        if l == '===END HERE':
            break
        print(l)
    elif l == '===START HERE':
        foundstart = True
