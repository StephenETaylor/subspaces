#!/usr/bin/env python3
"""
    actually, because I use the @ operator, I need python3.5 or better...

    See subsp9.py for previous history.  
    
    subspC.py tries a new idea:
    The pairs in the family relation are all gender-coded.  Some of them are 
    nearest neighbors, and susceptible to solving an analogy by similarity,
    instead of semantic combination, but all of them have ordered gender-role
    coding.  The 23 pairs could generate 23**2 difference vectors, all of them
    supposedly gender-coded, and randomly picking a pair will almost certainly
    not produce a gender-coded pair.  So if we hold out 6-10 pairs for testing,
    we have m = 169-289 difference vectors from the training pairs, which will 
    have both gender and non-gender components, and we should select 2m or 3m
    non gender vectors to pin. (we don't care if the underlying word vectors 
    move, so long as the difference vectors stay the same.)

    Hopefully the mean of the 200 or so training vectors is close to a sensible
    gender-coding, if we compute pca on the training vectors, 
    then try with gradually increasing dimension to find an additional transform
    which (as before) lines up the training vectors, but keeps the pinned 
    vectors constant (in the low dimension.) If we apply the alignment transform
    second, it has d**2 coefficients, while the training and pinned vectors
    have ~600d equations.  So the alignment transform is rather overdetermined.
    as we gradually increase d, we hope to watch the alignments of the 
    of the h**2 held-out test vectors match the trained alignment, that is the
    mean of the training vectors, while non-gender-coded pairs do not.
    In this stab, I'm skipping the word-analogy test for the moment


    subsp9.py, nov 30, added 'spare' baseline statistic
    subsp9.py adds regularization

    This code directly based on subsp4.py, but uses a new training objective.
    subsp4.py trains to change the difference vector.  Instead I want to 
    move the two halves of the pair, each halfway to the mean difference vector,
    so that we have twice as many coefficients to train with; 
     on first pass at this code, I added mean/2 to first of pair,
     subtracted mean/2 from second.  points were then quite close, but that
     doesn't really do what I meant to.
     result was that I got 100% results on training set, 0 on everything else

    On this pass, I want to think of a parametric line drawn with 
     (start vector) + s*(mean vector).  For example the line determined by
     vectors A and B is A+(B-A)*s, and I want to move A -> A', the point
     on the line (A+B)/2 + Mean*s closest to A; similarly for B
     Mean is determined from all the pairs of the training set.

    Define C = (A+B)/2
    Then A' = C + M*s_0 where s_0 is the solution to s in d EuclideanDist/ds = 0
     EuclideanDist = (C+M*s-A) .dot (C+M*s-A)
     d EuclideanDist/ds = 0 = 2(C+M*s-A)(M)
                          0 = (C-A) .dot (M) + (M).dot(M)*s
                          |M|**2 s = -(C-A) .dot (M)
                          s_0 = (A-C) .dot(M))/|M|**2
                              = 0.5*(A-B).dot(M)/M.dot(M)

     and B' = C - M * s_0


     Idea to try:  a number of 'pinned' items, outside the relation.  For these
     random (thus very likely outside the relation) items, the target is the 
     current location. (subsp6.py)
    
    and I want, perhaps in a followup, (this is that followup) (subsp7.py)
    to follow Tomas scheme of training both y = x@C and x = y@C.T,
    which should produce a transformation C@C.T ~= I; that is:
    1) the C matrix is (almost) non-singular, and C.T is (almost) it's inverse.
    2) since the off-diagonal elements of C@C.T are (almost) zero,
        the dot product of column i of C with row j of C.T is 
        (almost) zero if i != j
        (almost) one if i == j, and row j of C.T is column j of C, hence
    3) rows (also columns) of C are (almost) orthogonal, i.e. cos = 0


    I think that should mean that most angles in the result should not change 
    much, but of course the whole purpose of the training is to change some
    angles!   

    Want to see how that affects statistics.

    Realized this morning that I can use matrix operations to speed up 
    training.  See comments in train code (subsp8.py)

    Here's what I've seen hints of:
    1) when I do a 300 dimensional linear transformation on the data to
       make the difference vectors of a relation parallel,
       A) The analogy success rate on the training pairs is 100%
       B) The observed angles of the held-out test pairs are less
       C) I hope B leads to better success on the test pairs

    so: for (1) I have two techniques,
                x) training with back-propagation
                w) using numpy.linalg.lstsq for exact solution
            (w) is much faster, but seems to 
                  create 90-180 degree angles between some test pairs 
                  (x) has maximum angles of 60 - 90 degrees.
        I'm going to use (x) here to produce the xform.
        I need to measure:  
           original success rate
              [for held-out pairs only?]
           success rate after xform 
              [for training pairs]
              [for held-out pairs only]

              angles are interesting, but not really important...except to 
              the paper abstract ... 


"""

import datetime
import math
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import os
import random
import stats
import sys
import vectorstuff as vs

# some of these adjusted by commandline processing main()
Relation = 'relations/family' # Relation to examine
Holdout = 8                          # number of pairs to reserve for testing
vectorfile = 'George.bin'      
Maxpairs = 150000                    # max pairs to examine in relation
LearningRate = 0.001
Iterations = 3001
Regularization = 0.99
Cwidth = 300
Pinned = 50                          # number of pinned-down 'negative' items

def neighbors(n,vec):
    """
        neighbors returns list of n nearest neighbors to vec, like vs.nearest()
        doesn't require data to be normalized.
        uses a cos*abs(cos) distance measure, 
            which is 1.0 correlated with cos dist
            and -1.0 correlated with Euclidean dist of unit normalized data,
            but doesn't require normalization
    """
    answer = []
    vecnorm = vec .dot(vec)
    wec = vec/math.sqrt(vecnorm)   # pre-normalize search vector

    #preload answer array with ten items (to avoid i<10 in main loop)
    x = enumerate(vs.data)
    for i,d in x:
        if i>=10:
            break
        dnorm = d .dot (d)
        d0 = wec .dot(d) 
        dist = d0*abs(d0)/dnorm
        answer.append((dist,vs.words[i]))

    for i,d in x:       
        # use a cos*abs(cos) distance measure to avoid the sqrt
        dnorm = d .dot (d)
        if dnorm == 0: 
            # if we don't encounter this printout, we can remove this 
            # check for a particular vectorfile
            print('item',i,vs.words[i], 'has zero norm')
            continue
        d0 = wec .dot(d) 
        dist = d0*abs(d0)/dnorm
        if dist > answer[0][0]:
            answer[0] = (dist,vs.words[i])
            answer.sort()
    return answer


def train(diffvec,nrel,target):
    """
        parameter diffvec is an array of difference vectors
        nrel is number of vector inthe array
        return a C array such that 
        E = diffvec @ C yields an array E == target
    """

    # See lengthy comment at top of program about training
    a = LearningRate   #0.0000001  # should this be a parameter?
    # initialize C in (-1,1).  Too broad?  Parameter?
    # a little tough to get the dtype and shape arguments to work...
    C = 2*np.reshape(npr.random_sample(300*Cwidth).astype(np.float32),(300,Cwidth)) - 1
    for shebang in range(Iterations):
        #E = np.ones(shape = (nrel,Cwidth, dtype=np.float32)
        E = diffvec@C  # matrix multiplication (works!)
        #Q = np.ndarray(shape=(nrel,Cwidth), dtype=np.float32)
        #bug:Q =  1-E*E  # element-wise operation: Q[i][j] = 1 - E[i][j]**2
        if shebang%100 == 0:
            # don't need to compute Q at all, just its partial derivative,
            # but want to report it every once in a while
            t = target-E # broadcast t[i][j] = target[i][j] - E[i][j]
            Q = t*t # this is elementwise, so Q[i][j] = (1-E[i][j])**2
            sss = np.sum(Q)
            print('goal:', sss ,C[0][0], shebang)
        #  partial d. Q_{ij} wrt C_{kj} = -2 (1-E_{ij}) * D_{ik}
        # partial d. C_{kj} wrt Q_{ij} = 1/(2*(E_{ij}-1)) / d_{ik}
        #Ct = C.T        # get transpose for below...

        # set up comparison array
        #dDelta = np.zeros(shape=(Cwidth,Cwidth), dtype=np.float32)
        #for i in range(nrel):
        #    for j in range(Cwidth):
        #        fac1 = 2*(E[i][j]-target[i][j])            #1/2/(E[i][j]-1)
                #for k in range(300): # C[k][j] changed i*j times in the nested
                                     # loops, but the changes are additions,
                                     # so the effect is the same as combining
                                     # the additions first.
                #    C[k][j] -= a*fac1*diffvec[i][k]  # was divide, -a...
        #       Ct[j] -= a*fac1*diffvec[i] # make that loop a vector operation
        #        dDelta[j] -= a*fac1*diffvec[i] # make that loop a vector operation
        Fac1 = 2*(E - target).T
        dC = a* Fac1 @ diffvec
        # moved this down:C -= dC.T
        #huh = dC+(dDelta)
        #print (np.sum(huh))
        #C += dDelta.T
        #C = Ct.T # then get back to untranposed form
        # following is the "orthogonalizing goal"
        # repeat the computation for x = y*C.t; I'm going just repeat it...

        if False: # omit orthogonalization constraint?
            E = target @ (C.T) # since Ct wasn't changed in the previous loop
            Fac1 = 2*(E-diffvec).T
            eC = a * Fac1 @ diffvec
            #C -= dC.T
            #C -= eC #.T
            C = (Regularization*C) - dC.T - eC
        else:
            C = (Regularization*C) - dC.T # don't train in orthogonalization, or regularize

        #dDelta = np.zeros(shape=(Cwidth,Cwidth), dtype=np.float32)
        #for i in range(nrel):
        #    for j in range(Cwidth):
        #        fac1 = 2*(E[i][j]-diffvec[i][j]) 
        ##       C[j] -= a*fac1*diffvec[i] # change coefficients
        #        dDelta[j] -= a*fac1*diffvec[i] # change coefficients
        #but C already transposed version of Ct
        pass
        
    return C


def user_interface():
    """
        very minimal user interface requires careful placement of parameters
        if subsp9 is imported as a module, the globals can be set in custom code
    """
    global Maxpairs, Relation, Holdout, Pinned, Iterations, Regularization
    global vectorfile
    #minimal command line  maxpairs relation holdout
    if len(sys.argv) > 1 and sys.argv[1] != '-' :
        Maxpairs = int(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] != '-' :
        Iterations = int(sys.argv[2])
    if len(sys.argv) > 3 and sys.argv[3] != '-' :
        Holdout = int(sys.argv[3])
    if len(sys.argv) > 4 and sys.argv[4] != '-' :
        Pinned = int(sys.argv[4])
    if len(sys.argv) > 5 and sys.argv[5] != '-' :
        Regularization = float(sys.argv[5])
    if len(sys.argv) > 6 and sys.argv[6] != '-' :
        Relation = (sys.argv[6])
    if len(sys.argv) > 7 and sys.argv[7] != '-' :
        vectorfile = (sys.argv[7])
    if len(sys.argv) > 8 and sys.argv[8] != '-' :
        LearningRate = float(sys.argv[8])

def main():
    # echo parameters:
    print(sys.argv[0], str(datetime.datetime.now()))
    
    print('Relation',Relation)
    print('Maxpairs',Maxpairs)
    print('Holdout',Holdout)
    print('Pinned',Pinned)
    print('Iterations',Iterations)
    print('Regularization',Regularization)
    print('vectorfile',vectorfile)
    print('LearningRate',LearningRate)

    # get vectorfile
    vs.binloadV(vectorfile,150000)
    
    #read in relation
    pairlist = []
    differencevec  = []#   np.array([0]*300, dtype=np.float32)
    targvec  = []
    pairvec = []
    with open(Relation) as fin:
        for i,lin in enumerate(fin):
            if i == Maxpairs: break
            pair = lin.strip().split()
            if len(pair) != 2:
                cry()
            # possibly make calls to vs.adjust()...
            pair[0] = vs.adjust(pair[0])
            pair[1] = vs.adjust(pair[1])
            v0 = vs.vecs.get(pair[0],None)
            if type(v0) == type(None):
                print('?unknown word',pair[0])
                continue
            v1 = vs.vecs.get(pair[1],None)
            if type(v1) == type(None):
                print('?unknown word',pair[0])
                continue
            pairvec.append(v0)
            pairvec.append(v1)
            pairlist.append(pair)
            dv = v1-v0 #vector elementwise subtraction
            differencevec .append( dv )
    n = len(pairlist)
    #mn =  differencevec[:-Holdout,:].sum(axis =0)*(1/(n-Holdout))   # mean difference vector 
    mn = np.ndarray(300,dtype=np.float32)
    mn = 0
    for x in differencevec[:-Holdout]:
        mn += x
    mn = (1/(n-Holdout))*mn

    pairvec = np.array(pairvec,dtype=np.float32)
    targvec = np.ndarray(shape=(2*n,300),dtype=np.float32) #?

    # fill up pinned array with a bunch of outside the relation (probably)
    # examples which should stay in one place
    # by using difference vectors between random items, we greatly
    # increase the likelihood that the difference vectors do not belong in 
    # the relation
    pinned = np.ndarray(shape=(Pinned,300), dtype = np.float32)
    for i in range(Pinned):
        j = random.randint(1000,40000)
        k = random.randint(1000,40000)
        pinned[i] = vs.data[j] - vs.data[k]

    #check to see if we already computed the baseline, don't repeat work
    # method:  file relation.bl will contain relation, # pairs\n bl array
    blfile = os.path.basename(Relation) + '.bl'
    if os.path.exists(blfile): # use exists instead of isfile to allow sym link
        with open(blfile) as fin:
            bl = []
            dbl = []
            for i,lin in enumerate(fin):
                line = lin.strip().split()
                if i == 0:
                    if int(line[1]) != n:
                        sys.stderr.write('?bah, humbug! wrong pair count in .bl file\n')
                        sys.stderr.write('count is ')
                        sys.stderr.write(line[1])
                        sys.stderr.write(' file is ')
                        sys.stderr.write(blfile)
                        sys.stderr.write('\n')
                        sys.exit(1)
                    continue
                if i == 2*n+1: # must be last line
                    spare = int(line[0]) 
                    break

                if len(line) != n:
                    sys.stderr.write('? line wrong length: ')
                    sys.stderr.write(str(len(line)))
                    sys.stderr.write(' instead of ')
                    sys.stderr.write(str(n))
                    sys.stderr.write('\n line: \n')
                    sys.stderr.write(lin)
                    sys.exit(1)
                if not i>n:
                    row = []
                    for st in line:
                        row.append(int(st))
                    bl.append(row)
                    continue
                if len(bl) != n:
                    sys.stderr.write('? bl file too short.  Need ')
                    sys.stderr.write(str(n))
                    sys.stderr.write('lines, but have ')
                    sys.stderr.write(str(len(bl)))
                    sys.stderr.write('\n')
                    sys.exit(1)
                # fill up dbl array
                row = []
                for st in line:
                    row.append(float(st))
                dbl.append(row)

    else:
        #compute baseline
        bl = [[0]*n for i in range(n)] #store baseline success (before xform)
        dbl = [[0]*n for i in range(n)] # distance to nearest neighbor, baseline
        spare = 0
        for i in range(n):
            v0 = vs.vecs[pairlist[i][0]]
            v1 = vs.vecs[pairlist[i][1]]
            d10 = v1 - v0
            for j in range(n):
                if i == j: continue
                v2 = vs.vecs[pairlist[j][0]]
                #v3 = vs.vecs[pairlist[j][1]]
                a3 = d10 + v2
                nn = neighbors(10,a3)
                dbl[i][j] = nn[-1][0]       # store distance to nearest neighbor

                #check if correct answer has the Mikolov 'haram' in front of it
                if nn[-1][1] == pairlist[j][1]:
                    bl[i][j] = 1
                elif (nn[-1][1] == pairlist[j][0] or nn[-1][1] == pairlist[i][0]
                    or nn[-1][1] == pairlist[i][1] ):
                    if nn[-2][1] == pairlist[j][1]:
                        spare += 1
                    elif (nn[-2][1] == pairlist[j][0] or 
                        nn[-2][1] == pairlist[i][0]
                        or nn[-2][1] == pairlist[i][1] ):
                        if nn[-3][1] == pairlist[j][1]:
                            spare += 1
                        elif (nn[-3][1] == pairlist[j][0] or 
                            nn[-3][1] == pairlist[i][0]
                            or nn[-3][1] == pairlist[i][1] ):
                            if nn[-4][1] == pairlist[j][1]:
                                spare += 1
                    


        # and cache baseline for next run.  (needed primarily in debugging...
        #  but also when doing multiple values of Holdout for graphs)
        blfile = os.path.basename(Relation) + '.bl'
        with open(blfile,'w') as fout:
            fout.write(Relation)
            fout.write(' ')
            fout.write(str(n))
            fout.write('\n')

            # write bl, dbl arrays
            for ar in [bl,dbl]:
                for i in range(n):
                    for j in range(n):
                        fout.write(str(ar[i][j]))
                        fout.write(' ')
                    fout.write('\n')
            fout.write(str(spare))
            fout.write('\n')

    # whether read from file or recomputed, announce baseline
    print ('total analogies', n*(n-1), 'tot base success', np.sum(bl)) 
           
    basesum = [[0]*2 for i in range(2)]
    basedst = [[0]*2 for i in range(2)]
    for i in range(n):
        if i>=n-Holdout:
            ti = 1
        else: 
            ti = 0
        #ti = 1 if i<Holdout else 0
        for j in range(n):
            if j>= n-Holdout: 
                tj = 1
            else:
                tj = 0
            #tj = 1 if j<Holdout else 0
            basesum[ti][tj] += bl[i][j]
            basedst[ti][tj] += dbl[i][j]


    #compute xform
    #D = npl.lstsq(diffvec[:Holdout],targvec[:Holdout],rcond=None)
    #C = D[0]
    items = n-Holdout+Pinned
    pp = np.ndarray(shape=(items,300), dtype = np.float32)
    tp = np.ndarray(shape=(items,300), dtype = np.float32)
    for i,j in enumerate(range(n-Holdout)):
        pp[i] = differencevec[i]
        tp[i] = mn #targvec[j]
    for i in range(Pinned):
        j = i+(n - Holdout)
        pp[j] = tp[j] = pinned[i]
    C = train(pp, items, tp)

    #C = train(pairvec[2*Holdout:],2*n-2*Holdout,targvec[2*Holdout:]) 

    #apply xform
    vs.xform(C)  # for nlp.lstsq, used C[0])

    # print base after training, but before checking, since checking is slow
    print('base case')
    for i in range(2):
        print(basesum[i][0],basesum[i][1])
    print('base dist')
    for i in range(2):
        print(basedst[i][0],basedst[i][1])

    # check all
    improve = [[0]*2 for i in [0,1]]
    worsen  = [[0]*2 for i in [0,1]]
    spares   = [[0]*2 for i in [0,1]]
    netchanges  = [[0]*2 for i in [0,1]]
    netchange =[[0]*n for i in range(n)] 
    for i in range(n):
        v0 = vs.vecs[pairlist[i][0]]
        v1 = vs.vecs[pairlist[i][1]]
        d10 = v1 - v0
        for j in range(n):
            if i == j: continue
            v2 = vs.vecs[pairlist[j][0]]
            #v3 = vs.vecs[pairlist[j][1]]
            a3 = d10 + v2
            nn = neighbors(10,a3)

            if i>=n-Holdout: ti = 1
            else:         ti = 0
            if j>=n-Holdout: tj = 1
            else:         tj = 0

            if ti==0 and tj == 1:
                tj = 1 # instead of pass.  Can I place a bkpoint on a pass?

            td = dbl[i][j]
            tn = nn[-1][0]
            tt = tn-td
            netchange[i][j] += tt
            #netchange[i][j] += nn[-1][0] - dbl[i][j]
            netchanges[ti][tj] += tt
            if nn[-1][1] == pairlist[j][1]:
                if bl[i][j] == 0:
                    improve[ti][tj] += 1
            else:
                if bl[i][j] == 1:
                    worsen[ti][tj] += 1

                #check if correct answer has the Mikolov 'haram' in front of it
                if nn[-1][1] == pairlist[j][1]:
                    bl[i][j] = 1
                elif (nn[-1][1] == pairlist[j][0] or nn[-1][1] == pairlist[i][0]
                    or nn[-1][1] == pairlist[i][1] ):
                    if nn[-2][1] == pairlist[j][1]:
                        spares[ti][tj] += 1
                    elif (nn[-2][1] == pairlist[j][0] or 
                        nn[-2][1] == pairlist[i][0]
                        or nn[-2][1] == pairlist[i][1] ):
                        if nn[-3][1] == pairlist[j][1]:
                            spares[ti][tj] += 1
                        elif (nn[-3][1] == pairlist[j][0] or 
                            nn[-3][1] == pairlist[i][0]
                            or nn[-3][1] == pairlist[i][1] ):
                            if nn[-4][1] == pairlist[j][1]:
                                spares[ti][tj] += 1

    print('total change distances')
    for i in range(2):
        for j in range(2):
            print(netchanges[i][j], end='  ')
        print()

    print('\nimprove')
    for i in range(2):
        print(improve[i][0], improve[i][1])

    print('\nworsen')
    for i in range(2):
        print(worsen[i][0], worsen[i][1])

    print('bspare', spare)
    print('\nspares')
    for i in range(2):
        print(spares[i][0], spares[i][1])
    print('finishing time:',str(datetime.datetime.now()))


if __name__ == '__main__':
    user_interface()
    main()



