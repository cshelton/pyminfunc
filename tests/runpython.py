#!/usr/bin/python

import subprocess
import sys

def runpython(testfile):
    output = subprocess.check_output(['python','runone.py',testfile]).decode('utf-8')
    foundstart = False

    out = ''
    for l in output.split('\n'):
        if foundstart:
            if l == '===END HERE':
                break
            out += l + '\n'
        elif l == '===START HERE':
            foundstart = True
    return out

if __name__ == '__main__':
    testfile = sys.argv[1]
    print(runpython(testfile),end='')
