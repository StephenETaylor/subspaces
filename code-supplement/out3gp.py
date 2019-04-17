#!/usr/bin/env python3
"""
    read a file $1.out and write a file $1.gp 
    this file reads a file produced by a casu.py run
"""

import sys

Filename = 'E'
Fields = Line = Lin = fout = fin =  Line = Lin = None
State = 0
nl = '\n'

def nextline():
    global Fields,  Lin, Line
    Lin = next(fin, None)
    if Lin == None: return None
    Line = Lin.strip().split(' ')
    Fields = Line.__iter__() 
    return Line

def nextf():
    if Fields == None:
        nextline()
    return next(Fields)

def pout(x):
    fout.write(x)

def pout2n():
    fout.write(Line[0]+' '+Line[1]+' ')
    nextline()

def p3l():
    pout(Line[0]+' ')
    nextline()
    pout2n()
    pout2n()


def main():
    global Filename, fout, fin, Line, Lin, State
    if len(sys.argv) > 1:
        Filename = sys.argv[1]

    with open(Filename+'.gp', 'w') as fout:
        with open(Filename+'.out') as fin:
            while True:
                if State == 0: 
                    if Lin == None:
                        nextline()
                    if Line[0] != 'casu.py' and Line[0] != 'casv.py':
                        sys.stderr.write('?Expected "casu.py", got:\n')
                        sys.stderr.write(Lin)
                        sys.exit(1)
                    pout('#'+Lin)
                    nextline()
                    State = 1
                elif len(Line) == 0 or Lin == '\n':
                    nextline()
                elif Line[0] == 'finishing':
                    State = 0
                    test = nextline()
                    if test == None:
                        break
                elif Line[0] == 'Relation':
                    pout('#'+Lin)
                    nextline()
                elif (Line[0] == 'Maxpairs' or 
                    Line[0] == 'Regularization' or 
                    Line[0] == 'goal:' or 
                    Line[0] == 'Iterations'):
                    nextline()
                elif (Line[0] == 'Holdout' or 
                    Line[0] == 'xgoal:' or 
                    Line[0] == 'Pinned' or
                    Line[0] == 'bspare' ):
                    pout2n()
                elif Line[0] == 'goal:':
                    nextline()
                elif (Line[0] == '?unknown' and Line[1] == 'word'):
                    nextline() # skip over announcement of OOV
                elif ((Line[0] == 'base' and Line[1] == 'dist') or
                    (Line[0] == 'total' and Line[1] == 'change')) :
                    nextline()
                    nextline()
                    nextline()
                elif (Line[0] == 'base' and Line[1] == 'case'):
                    pout2n()
                    pout2n()
                    pout2n()
                elif (Line[0] == 'improve' or
                    Line[0] == 'worsen' or
                    Line[0] == 'spares'):
                    p3l()
                elif (Line[0] == 'total' and Line[1] == 'analogies'):
                    pout('totalanalogies ')
                    pout(Line[2])
                    pout(' basesuccess ')
                    pout(Line[6]+' ')
                    nextline()
                else:
                    sys.stderr.write('?Failed to process line:\n')
                    sys.stderr.write(Lin)
                    nextline()


if __name__ == '__main__':
    main()
