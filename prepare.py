#!/usr/bin/ python
import numpy as np
from explore import dbraw, key

# keys for discrete classes
keydisc = []
k1 = list(range(1,11))
k2 = [2,4]
for tooth in key[1:10]:
    for item in k1:
        string = tooth + ' ' + str(item)
        keydisc.append(string)
for item in k2:
    string = key[10] + ' ' + str(item)
    keydisc.append(string)

# dimensionalizing classes, unneeded
db = {}
for toothdisc in keydisc:
    db[toothdisc] = []
    for tooth in key[1:]:
        if toothdisc.split()[:-1] == tooth.split():
            for item in dbraw[tooth]:
                if float(toothdisc.split()[-1]) == item:
                    db[toothdisc].append(1)
                else:
                    db[toothdisc].append(0)

# creating matrix from dict
def buildX(db, key):
    X = np.zeros( ( len(db[key[0]]) , len(key) ) )
    i = 0
    for tooth in key:
        X[:, i] = db[tooth]
        i = i + 1
    return X

# randomizing and splitting data
def randsplit(X):
    from random import shuffle
    N = X.shape[0]
    F = X.shape[1]
    idx = list(range(N))
    shuffle(idx)

    # splitting samples into train/test
    ntrain = int(0.75*N) 
    ntest  = N - ntrain

    # initialize
    Xtrain = np.zeros((ntrain,F))
    Xtest  = np.zeros((ntest ,F))

    # populate
    for n in range(N):
        if n < ntrain:
            Xtrain[n,:] = X[idx[n],:]
        else:
            Xtest[n-ntrain,:] = X[idx[n],:]
    return Xtrain, Xtest

