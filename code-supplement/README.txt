This directory contains programs used in the experiments described
in the paper: Is there a linear subspace in which difference vectors
for word analogy pairs are parallel?

The program used for most of the experiments is subspB.py, which 
accepts an input relation and a word embedding, splits the relation  into a training and test set,
and generates a linear transformation for the embedding such that the 
difference vectors for the pairs in the training set are close to parallel.
It then prints baseline and improvement statistics for analogies of four kinds,
created from the pairs of the relation:  
trained pair :: trained pair;
untrained pair :: trained pair;
trained pair :: untrained pair;
untrained pair :: untrained pair.

Other files shown provide support for this program, or for preparing
figures or tables.




Descriptions of individual files
Bu.py
    this program is imported by subspB.py.  It is used to assist with Arabic
    script.  The experiments were done first with Arabic morphological
    relations.

casu.py
casv.py
    These two programs import subspB.py, and run the main module with different
    values of the parameters.

g9b.gnuplot
    input file to prepare a pdf file of plots from a .g9 file, as prepared by
    out2g9.py from a transcript of a subsp9.py run.

Google-analogy-dataset
    This directory contains files of 19-40 pairs.  Each file of N pairs can be
    expanded into N*N-1 analogies by coupling any pair with any other in the
    file.

GSK300.bin
    this file is a binary file in a format used by the vectorstuff.py module
    It is a version of GoogleNews-vectors-negative300.bin, which can be 
    downloaded from https://code.google.com/arcive/p/word2vec

out2g9.py
out3gp.py
    produce slightly different output files from a subsp*.py transcript
`
plot.py
plotting.py
    a pair of programs which together produce the rosettes in figure 1.

readw2v.py
    This program reads a word2vec format word embedding file,
    and outputs a subset in a format suitable to the vectorstuff module

stats.py
    This module contains a python class which accumulates datapoints, and
    computes mean, standard deviation, skew for them.

subspB.py
    The central program used in the final version of the paper.  As the 
    'B' indicates, it is heir to a series of previous experiments.

vectorstuff.py
    A python module for dealing with word-embedding files.  Contains code
    for reading and writing text and binary files, vector normalization,
    nearest neighbors to a vector...
