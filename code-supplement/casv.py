#!/usr/bin/env python3.5
"""
   read subsp9 with import, instead of redoing user interface
"""
import subsp9
import sys

relations = [ 'relations/capital-common-countries', 'relations/capital-world',
	 'relations/city-in-state', 'relations/country-currency', 
	'relations/family', 'relations/gram1-adjective-adverb' ,
	'relations/gram2-opposite' ,'relations/gram3-comparative' ,
	'relations/gram4-superlative' ,'relations/gram5-present-participle' ,
	'relations/gram6-nationality-adjective' ,'relations/gram7-past-tense' ,
	'relations/gram8-plural' ,'relations/gram9-plural-verbs']

subsp9.vectorfile = 'George.bin' #'../GoogleNews-vectors-negative300.bin'
subsp9.Pinned = 50
subsp9.Holdout = 8
subsp9.Regularization = 0.98
subsp9.LearningRate = 0.001


if len(sys.argv) > 1:   first = int(sys.argv[1])
else:                   first = 0
if len(sys.argv) > 2:   last  = int(sys.argv[2])
else:                   last =  first+1 #only one relation file
for x in relations[:1]:
#for x in relations[:3]:
    subsp9.Relation = x
    for h in range(7):
       subsp9.Holdout = 4+h*2
       for p in [50,100,150,200,250,500]:
            subsp9.Pinned = p
            subsp9.main()
