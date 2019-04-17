#!/usr/bin/python3

"""
    read a word2vec format vector file, write out a vector file in my .bin 
    format.
"""

import numpy as np
import sys

Maxvecs=150000
Words = Vectors = None

def readword2vec(filename, Maxvecs):
    """
        read at most Maxvecs from a word2vec binary file named filename
        format of the file:
           first line:  <number of vectors> <width of vectors>
           all other lines: <space-terminated word> <width-of-vectors float32s>
    """
    global Words, Vectors
    with open(filename,'rb') as bf:
        first = bf.readline().decode('utf8')
        line = first.strip().split()
        filemaxvecs = int(line[0])
        filewidth = int(line[1])
        if int(line[1]) != 300:
            sys.stderr.write('?funny row width: ')
            sys.stderr.write(line[1])
            sys.stderr.write('?for file:')
            sys.stderr.write(filename)
            sys.exit(1)
        Vectors = np.ndarray(shape=(Maxvecs,1200),dtype=np.uint8)
        Words = [None]*Maxvecs
        buffe = bytearray(4096) #np.ndarray(shape=4096, dtype=np.uint8)
        bf.readinto(buffe)
        for i in range(Maxvecs):
            ends = buffe.find(32)
            while ends < 0:
                s = bytearray(4096)
                l = bf.readinto(s)
                buffe += s
                ends = buffe.find(32)
            Words[i] = buffe[:ends].decode('utf8')
            buffe = buffe[(ends+1):] #skip the space...

            while len(buffe) < 1200:
                s = bytearray(4096)
                l = bf.readinto(s)
                buffe += s
            Vectors[i] = buffe[:1200]
            buffe = buffe[1200:]

def savebinaryfile(fn):
    with open(fn,'wb') as of:
        of.write(fn.encode('utf-8'))
        of.write(b'\n')
        of.write(str(Maxvecs).encode( encoding = 'utf-8'))
        of.write(b'\n')
        for i in range(Maxvecs): # save words array. reconstitute voc onload
            of.write(Words[i].encode( encoding = 'utf-8'))
            of.write(b'\n')
        for i in range(Maxvecs): # save vectors
            Vectors[i].tofile(of)
                                                                    

                    
readword2vec(sys.argv[1],Maxvecs)
savebinaryfile('George.bin')





