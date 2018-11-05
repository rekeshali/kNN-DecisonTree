#!/bin/usr/python
import numpy as np
from explore import dbraw as db, key
from prepare import buildX, randsplit
from implement import kNN, GenerateTree, TreeClassify, perform, metrics
from output import kNNout, TreeOut
#############################################################################
############# kNN ###########################################################
#############################################################################
# # klist     = [4,8]
# # Lnormlist = [2,5]
# klist     = list(range(2,9)) + [ 17, 33]
# Lnormlist = list(range(1,11));
# TF        = np.zeros(( len(Lnormlist), 4, len(klist) ))
# M         = np.zeros(( len(Lnormlist), 5, len(klist) ))
# 
# X = buildX(db, key[1:])
# iters = 100; 
# for ki, k in enumerate(klist):
#     for li, Lnorm in enumerate(Lnormlist):
#         for i in range(iters):
#             # randomize and split
#             [Xtrain, Xtest] = randsplit(X)
#             # get respective class
#             [Ctrain, Ctest] = [ Xtrain[:,-1], Xtest[:,-1] ]
#             # run kNN on truncated data
#             CkNN = kNN(k, Lnorm, Xtrain[:,:-1], Xtest[:,:-1], Ctrain)
#             # get performance
#             TF[li,:,ki] += perform(CkNN,Ctest)
#         M[li,:,ki] = metrics(TF[li,:,ki])
# kNNout(klist, Lnormlist, iters, TF, M)
#############################################################################
############## Decision Tree ################################################
#############################################################################
entmax   = 0.1
depthmax = 40
dtypes   = [ 'discrete', 'numeric' ]
dtype    = dtypes[0]
imptypes = [ 'entropy', 'gini', 'misclassification error' ]
imptype  = imptypes[1]
TF       = np.zeros(( 4 ))
M        = np.zeros(( 5 ))
X        = buildX(db, key[1:])
iters = 100
for i in range(iters):
    [Xtrain, Xtest] = randsplit(X)
    Ctest = Xtest[:,-1]
    Ctree = []
    [depth, T] = GenerateTree(Xtrain, {}, imptype, entmax, 0, 0, depthmax, dtype)
    for row in Xtest:
        Ctree.append(TreeClassify(row, T, dtype))
    TF += perform(np.asarray(Ctree), Ctest)
M  = metrics(TF)
#############################################################################
############### END #########################################################
#############################################################################
