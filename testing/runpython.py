#!/usr/bin/python

import subprocess
import sys

testfile = sys.argv[1]
output = subprocess.check_output(['python','runone.py',testfile]).decode('utf-8')

foundstart = False


for l in output.split('\n'):
    if foundstart:
        if l == '===END HERE':
            break
        print(l)
    elif l == '===START HERE':
        foundstart = True
