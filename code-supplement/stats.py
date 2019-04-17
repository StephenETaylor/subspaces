#!/usr/bin/env python3
"""
    A class for accumulating min, max, mean and variance 
"""

import math

class stats:
    def __init__(self,statid = ''):
        self.statid = statid
        self.minval = None
        #minarg
        self.maxval = None
        #maxarg
        self.sumstat = 0
        self.sum2stat = 0
        self.sum3stat = 0
        self.sum4stat = 0
        self.itemcount = 0
        self.skew = self.kurtosis = -53

    def newitem(self,val,arg):
        """
            val is the value we are accumulating statistics on, and
            arg is an identifier used in print(), 
            so we'll know for which we see the min and max
        """
        self.itemcount += 1
        if type(self.minval) == type(None):
            self.minval = val
            self.minarg = arg
        elif self.minval > val:
            self.minval = val
            self.minarg = arg
        if type(self.maxval) == type(None):
            self.maxval = val
            self.maxarg = arg
        elif self.maxval < val:
            self.maxval = val
            self.maxarg = arg
        self.sumstat += val
        v2 = val*val
        v3 = v2*val
        v4 = v2*v2
        self.sum2stat += v2
        self.sum3stat += v3
        self.sum4stat += v4

    def compute(self):
        n = self.itemcount
        if n == 0: return
        mean = self.meanstat = self.sumstat /n
        mean2 = mean * mean
        varstat = self.sum2stat /n - mean2
        self.stddev = math.sqrt(varstat)

        mean3 = mean2 * mean
        mean4 = mean2 * mean2
        if self.stddev != 0:
            self.skew = (self.sum3stat/n -3*mean*self.sum2stat/n + 2* mean3)/self.stddev**3
            self.kurtosis = (self.sum4stat/n -4*mean*self.sum3stat/n + 6*mean2*self.sum2stat/n - 3*mean4)/self.stddev**4

    def print(self):
        # display current statistics on console
        self.compute() # compute is idempotent, so it's okay if it was called before
        print(self.statid+': min', self.minval, self.minarg, 'mean',self.meanstat, self.stddev, 'max', self.maxval, self.maxarg)

if __name__ == '__main__': # short test / sanity sequence
    import random

    thirty = 30
    s = stats('uniform')
    for i in range(thirty):
        s.newitem(random.uniform(0,180),i)
    s.print()
    print(s.skew, s.kurtosis)
    t = stats('gaussian')
    for i in range(thirty):
        t.newitem(random.gauss(90,30),i)
    t.print()
    print(t.skew, t.kurtosis)
