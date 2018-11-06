#!/bin/usr/python
import numpy as np
from explore import dbraw as db, key
from prepare import buildX, randsplit
from implement import kNN, GenerateTree, TreeClassify, PCA, perform, metrics
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
entmaxs  = [0.1, 0.2, 0.3, 0.4]
entmax   = entmaxs[0]
depthmaxs= [2,3,4,5,7,10]
depthmax = depthmaxs[0]
dtypes   = [ 'discrete', 'numeric' ]
dtype    = dtypes[0]
imptypes = [ 'entropy', 'gini', 'misclassification error' ]
imptype  = imptypes[1]
# TF       = np.zeros(( ] ))
# M        = np.zeros(( 5 ))
X        = buildX(db, key[1:])
# Xk = PCA(X[:,:-1], 'kmin', 0.9)
# Xk = np.hstack((Xk,X[:,-1].reshape(X.shape[0],1)))
# X = Xk
depth = np.zeros( (len(depthmaxs), len(dtypes), len(imptypes), 2) )
depth[:,:,:,1] = float('inf')
TF    = np.zeros( (len(depthmaxs), len(dtypes), len(imptypes), 4) )
M     = np.zeros( (len(depthmaxs), len(dtypes), len(imptypes), 5) )
iters = 100
for ei, depthmax in enumerate(depthmaxs):
    for di, dtype in enumerate(dtypes):
        for ii, imptype in enumerate(imptypes):
            for i in range(iters):
                [Xtrain, Xtest] = randsplit(X)
                Ctest = Xtest[:,-1]
                Ctree = []
                [depthtemp, T] = GenerateTree(Xtrain, {}, imptype, entmax, 0, 0, depthmax, dtype)
                if depthtemp > depth[ei,di,ii,0]:
                    depth[ei,di,ii,0] = depthtemp
                if depthtemp < depth[ei,di,ii,1]:
                    depth[ei,di,ii,1] = depthtemp
                for row in Xtest:
                    Ctree.append(TreeClassify(row, T, dtype))
                TF[ei, di, ii, :] += perform(np.asarray(Ctree), Ctest)
            M[ei, di, ii, :]  = metrics(TF[ei, di, ii, :])
TreeOut(depthmaxs, dtypes, imptypes, iters, TF, M, depth)
#############################################################################
############### END #########################################################
#############################################################################
