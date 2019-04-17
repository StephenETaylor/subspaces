#!/usr/bin/env python3
"""
    accepts a vector file and a pathname to a relation, and 
    produces a plot file.

"""

import math
import numpy as np
from scipy import linalg as LA
import os.path
import plotting
import stats
import sys
import vectorstuff as vs

Flashy = False #True   # redraw screen frequently
Rosette = False # plot rosette of 2-D difference vectors
Field = False # plot field of difference vectors
PairPCA = False #use only pairs in pairfile for PCA
PrunePause = False # pause after each pruning of a pair
Final = False #display at end of pruning loop, not after each pruning
ShowMean = False #plot mean on Rosette view
Histogram1 = None# print list of 2-D angles with X-axis for use as histogram
Histogram2 = None# print list of angles with mean vector for use as histogram
Histogram3 = None# print list of angles with [1]*300 for use as histogram
Verbose = True# print lots of stuff
maxvecs = 30000
PCAfile = None
HackPCA = 0

vectorsFile = "/home/staylor/fall17s/CrossLingualSemanticSpaces/data/vectors/wiki.ar.vecTomas"
#vectorsFile = "20170920.vec"

pairfiles = [
'relations/vheher-vhehim.txt',
'relations/vheher-vsheher.txt',
'relations/vheher-vshehim.txt',
'relations/vheher-vshe.txt',
'relations/vhehim-vsheher.txt',
'relations/vhehim-vshehim.txt',
'relations/vhehim-vshe.txt',
'relations/vhe-vheher.txt',
'relations/vhe-vhehim.txt',
'relations/vhe-vsheher.txt',
'relations/vhe-vshehim.txt',
'relations/vhe-vshe.txt',
'relations/vsheher-vshehim.txt',
'relations/vshe-vsheher.txt',
'relations/vshe-vshehim.txt',
]

pairfile = pairfiles[0]
dumpfile = '/dev/null'
pruneCount = 0


def commandLine(sysvec):
    global Flashy, Rosette, Field, PrunePause, Final, ShowMean
    global Histogram1, Histogram2, Histogram3, PairPCA
    global Verbose, maxvecs, PCAfile, HackPCA
    global vectorsFile, pairfile, pruneCount, dumpfile
    skipArg = True
    for i,arg in enumerate(sysvec):
        if skipArg:
            skipArg = False
            continue
        elif arg == '-dump':
            dumpfile = sysvec[i+1]
            skipArg = True
        elif arg == '-field':
            Field = True
        elif arg == '-flashy':
            Flashy = True
        elif arg == '-hackPCA':
            HackPCA = float(sysvec[i+1])
            skipArg = True
        elif arg == '-histogram1':
            Histogram1 = sysvec[i+1]
            skipArg = True
        elif arg == '-histogram2':
            Histogram2 = sysvec[i+1]
            skipArg = True
        elif arg == '-histogram3':
            Histogram3 = sysvec[i+1]
            skipArg = True
        elif arg == '-maxvecs':
            maxvecs = int(sysvec[i+1])
            skipArg = True
        elif arg == '-noverbose':
            Verbose = False
        elif arg == '-pairfile':
            pairfile = sysvec[i+1]
            skipArg = True
        elif arg == '-pairpca':
            PairPCA = True
        elif arg == '-PCAfile':
            PCAfile = sysvec[i+1]
            skipArg = True
        elif arg == '-prune':
            pruneCount = int(sysvec[i+1])
            skipArg = True
        elif arg == '-prunepause':
            PrunePause = True
        elif arg == '-rosette':
            Rosette = True
        elif arg == '-showmean':
            ShowMean = True
        elif arg == '-final':
            Final = True
        elif arg == '-vecfile':
            vectorsFile = sysvec[i+1]
            skipArg = True
        else:
            print ("""usage:
            python3 plot.py <options>
                        -field                display difference vector field
                        -final                only show field/rosette at end
                        -flashy               display, pause for each pair
                        -hackPCA              gimmicky idea to hack PCA
                        -histogram1 filename  list of 2-D angles with x-axis
                        -histogram2 filename  list of angles with mean vector
                        -maxvecs number       use only number words from vector file
                        -pairfile filename    choose pairfile
                        -pairspca             use only pairfile for PCA calculation
                        -PCAfile              file of words to use for PCA
                        -prune integer        prune integer 'worst' pairs
                        -prunepause           pause after each pruning
                        -rosette              display difference vector rosette
                        -dump filename        store (pruned?) pairs in filename
                        -showmean             plot projection of mean on rosette
                        -vecfile filename     set vector file
                        -verbose              lots of outputs
""")
            sys.exit(1)



def addcoords(paircoords,w):
    t = vs.vecs.get(w,None)
    if type(t) != type(None):
        paircoords.append(t)
    else:
        sys.stderr.write('?PCA word outside vocabulary: ')
        sys.stderr.write(w)
        sys.stderr.write('\n')

def main():
    global pruneCount
    commandLine(sys.argv)
    
    print(pairfile,vectorsFile)

    #check for a binary vector file
    if vectorsFile[-4:] == '.bin' :
        vs.binloadV(vectorsFile)
    elif os.path.isfile(vectorsFile+'.bin'):
        vs.binloadV(vectorsFile+'.bin')
    else: # no binary file, load from ascii.
        vs.getvecs(vectorsFile,maxvecs)
        vs.binSaveV(vectorsFile)        #  but save binary file for next time

    data = vs.data

    # do pca on 300 dimensional data array to get two dimensional transform
    # for graphing
    if not PairPCA and type(PCAfile) == type(None):
        # debugging code commented out:
        #for (i,v) in enumerate(data): # check data for NaN
        #   for (j,n) in enumerate(v):
        #     if math.isnan(n): print(i,j,n)

        _,  evals, evecs = plotting.PCA(data,2)
        # now data[x].dot(evecs) [or vs.vecs[word].dot(evecs)] is a two-d coordinate


    # go through pairfile and save pairs 
    pairlist = list()
    paircoords = list()
    with open(pairfile) as fin:
        for lin in fin:
            slin = lin.rstrip().split('\t')
            if len(slin) != 2:
                gripe()
            pairlist.append((slin[0],slin[1]))
            if PairPCA:
                addcoords(paircoords,slin[0])
                addcoords(paircoords,slin[1])

    if type(PCAfile) != type(None):
        with open(PCAfile) as pin:
            for lin in pin:
                slin = lin.rstrip().split('\t')
                addcoords(paircoords,slin[0])

    if len(paircoords) > 0:
        _,  evals, evecs = plotting.PCA(np.array(paircoords),2)


    # go through evecs and hack them into binary vector
    if HackPCA > 0:
        for r in range(300):
            for c in range(2):
                if abs(evecs[r][c]) > HackPCA:
                    pass #evecs[r][c] = 1.0*evecs[r][c]
                else:
                    evecs[r][c] = 0
    # go through pairfile and rescale for the image
    xsize = stats.stats('xs')
    ysize = stats.stats('ys')
    dxsize = stats.stats('dx')
    dysize = stats.stats('dy')
    for slin in pairlist:
            # plot line from slin[0] to slin[1]
            v0 = vs.vecs.get(slin[0],None)
            v1 = vs.vecs.get(slin[1],None)
            if type(None) == type(v0) or type(None) == type(v1): continue
            vp0 = v0.dot(evecs)
            vp1 =v1.dot(evecs)
            xsize.newitem(vp0[0],slin[0])
            xsize.newitem(vp1[0],slin[1])
            ysize.newitem(vp0[1],slin[0])
            ysize.newitem(vp1[1],slin[1])
            #image.rescale((vp0[0], vp0[1]))
            #image.rescale((vp1[0], vp1[1]))
            dxsize.newitem(vp1[0]-vp0[0],slin)
            dysize.newitem(vp1[1]-vp0[1],slin)
            #dimage.rescale((vp1[0]-vp0[0],vp1[1]-vp0[1]))
            

    while pruneCount >= 0:

        # set up file to build image
        image = plotting.p(1024)
        #set scale, based on min and max values
        image.rescale((xsize.minval,ysize.minval))
        image.rescale((xsize.maxval,ysize.maxval))
        # similar file for difference vector rosette
        dimage = plotting.p()
        dimage.rescale((dxsize.minval,dysize.minval))
        dimage.rescale((dxsize.maxval,dysize.maxval))

        dvs = []   

        anglestats = stats.stats('angles')
        cosstats = stats.stats('cosins')
        mcosstat = stats.stats('meancos')
        mvAstat = stats.stats('meanAng')
        if Histogram3:
            danstat = stats.stats('diagAng')
            dangles = []

        angles = []
        mvAngles = []

        #find the mean difference vector
        sv = None
        for slin in pairlist:
            v0 = vs.vecs.get(slin[0],None)
            v1 = vs.vecs.get(slin[1],None)
            if type(None) == type(v0) or type(None) == type(v1): continue
            dv = v1-v0  # 300 dim difference vector
            if type(sv) == type(None):
                sv = dv
                scount = 1
            else:
                sv += dv
                scount += 1
        mv = sv / scount          #  compute mean vector

        # plot difference vector
        if ShowMean:
            dp = mv.dot(evecs)
            dimage.foreground = plotting.green
            dimage.drawline((0,0),(dp[0],dp[1]))

        if Histogram3:
            diag = np.array([1.0]*300)


        # now go though pairfile again and gather statistics
        for si,slin in enumerate(pairlist):
            # plot line from slin[0] to slin[1]
            v0 = vs.vecs.get(slin[0],None)
            v1 = vs.vecs.get(slin[1],None)
            if type(None) == type(v0) or type(None) == type(v1): continue
            rank0 = vs.voc.get(slin[0], vs.maxvecs+1)
            rank1 = vs.voc.get(slin[1], vs.maxvecs+1)
            p0 = v0 .dot (evecs)
            p1 = v1 .dot (evecs)
            image.drawline(p0,p1)
            if Verbose:
                print(si, rank0,slin[0]  ,rank1,slin[1] ,'',end = '')
            # p1-p0 is the difference vector.  We hope difference vectors
            # will turn out to be parallel, so we track the angles
            dvec = p1-p0
            dv = v1 - v0
            mcos = cosdist(mv, dv) # cos dist this difference from mean diff vec
            mvA = 180*math.acos(mcos)/math.pi
            mvAngles.append(mvA)
            if Histogram3:
                diagcos = cosdist(dv , diag)
                dangle = 180*math.acos(diagcos)/math.pi
                danstat.newitem(dangle,si)
                dangles.append(dangle)
            mvAstat.newitem(mvA,si)
            mcosstat.newitem(mcos,si)
            

            # we compute both atan2 and acos(dotproduct)
            # which can differ by sign, since principal values of 
            #  atan2 run from -pi .. +pi
            #  acos  run from 0 .. pi.  For cosin, cos (-x) = cos(x)
            ac = math.atan2(dvec[1],dvec[0])*180/math.pi 

            # compute dot product of x-axis with p1-p0
            ac3 = dvec[0]/math.sqrt(dvec[0]*dvec[0]+dvec[1]*dvec[1])
            ac4 = math.acos(ac3)*180/math.pi
            cosstats.newitem(ac3,lin)
            # 

            #print (ac4,ac)
            angles.append(ac)
            anglestats.newitem(ac,lin)

            dimage.drawline((0,0),dvec)
            if Verbose: print(mcos, ac)
            if Flashy:
                if Field:
                    image.show()
                if Rosette:
                    dimage.show()
                input() # await permission to continue

        if Histogram1:
            angles.sort()
            #print(angles)
            with open(Histogram1,'w') as fo:
                for x in angles:
                    fo.write(str(x))
                    fo.write('\n')
            anglestats.print()
            print(anglestats.kurtosis)
        if Histogram2:
            mvAngles.sort()
            #print(mvAngles)
            with open(Histogram2,'w') as fo:
                for x in mvAngles:
                    fo.write(str(x))
                    fo.write('\n')
            mvAstat.print()
            print(mvAstat.kurtosis)
        if Histogram3:
            dangles.sort()
            with open(Histogram3,'w') as fo:
                for x in dangles:
                    fo.write(str(x))
                    fo.write('\n')
            danstat.print()
            print(danstat.kurtosis)
        if (not Flashy) and (not Final):
            if Field:
                image.show()
            if Rosette:
                dimage.show()
        if Verbose: mcosstat.print()

        pruneCount += -1
        if pruneCount >= 0:
            del(pairlist[mcosstat.minarg]) #remove item with lowest cosine
                                           # similarity to mean
            if PrunePause:
                input() # await permission to continue
            
    if Final:
        if Field:
            image.show()
        if Rosette:
            dimage.show()
        mcosstat.print()
    # write possibly pruned pairfile to dumpfile (or to /dev/null)
    with open(dumpfile,'w') as fout:
        for slin in pairlist:
            fout.write(str(slin[0]))
            fout.write('\t')
            fout.write(str(slin[1]))
            fout.write('\n')
        fout.close()

    return image

# take cosine difference between two possibly unnormalized vectors
def cosdist(v, w):
    num = v .dot  (w)
    normv2 = v .dot (v)
    normw2 = w .dot (w)
    return num / math.sqrt(normv2 * normw2)



main()


