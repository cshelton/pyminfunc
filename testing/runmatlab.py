#!/usr/bin/python

import subprocess
import sys

def runmatlab(testfile):

    output = subprocess.check_output(['matlab','-nodesktop','-nojvm','-nosplash','-r','try, runone(\''+testfile+'\'); end; quit']).decode('utf-8')



    ret = ''
    for l in output.split('\n'):
        if foundstart:
            if l == '===END HERE':
                break
            ret += l + '\n';
        elif l == '===START HERE':
            foundstart = True
    return ret

def runmatlabcont(testfile,sp=None):

    if sp==None:
        sp = subprocess.Popen(['matlab','-nodesktop','-nojvm','-nosplash'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,universal_newlines=True,text=True,bufsize=1)

    sp.stdin.write('runone(\''+testfile+'\',0);\n')
    ret = ''
    foundstart = False
    for l in sp.stdout:
        if foundstart:
            if l == '===END HERE\n':
                break
            ret += l
        elif l == '===START HERE\n' or l == '>> ===START HERE\n':
            foundstart = True
    return ret,sp


if __name__ == '__main__':
    testfile = sys.argv[1]
    print(runmatlab(testfile),end='')
