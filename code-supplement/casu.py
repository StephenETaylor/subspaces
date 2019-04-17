#!/usr/bin/env python3.5
"""
   read subsp9 with import, instead of redoing user interface
"""
import subsp9

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

for x in relations:
	subsp9.Relation = x
	subsp9.main()
