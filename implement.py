import numpy as np
#######################################################################################
############################## K Nearest Neighbors ####################################
#######################################################################################
def kNN(k, Lnorm, Xtrain, Xtest, Ctrain):
    # compares all test data to all training data
    # nearest neighbors give probability of class
    [Ntrain, F] = Xtrain.shape       # samples in train
    Ntest       = Xtest.shape[0]     # samples in test
    dist        = np.zeros((Ntrain)) # dist of nts compared to ntr
    CkNN        = np.zeros((Ntest )) # class of nts according to kNN
    Clist       = []                 # list of all class values

    for c in Ctrain: # populate class list
        if c not in Clist:
            Clist.append(c)

    for nts in range(Ntest):
        for ntr in range(Ntrain):
            dist[ntr] =  (np.sum(np.abs(Xtest[nts,:] - Xtrain[ntr,:])**Lnorm))**(1/Lnorm)
        kNNidx = np.argsort(dist)[:k]  # get first k sorted distance indices
        Cprob = np.zeros((len(Clist))) # reinitialize to zero probability
        for nn in kNNidx: # for all k neighbors
                Cidx = Clist.index(Ctrain[nn]) # find index of class in list
                Cprob[Cidx] += 1 # add one to sum at index of class 
        Cprob /= k # divide by k to get prob
        maxprob = np.max(Cprob) # find max
        CkNN[nts] = Clist[list(Cprob).index(maxprob)] # class = first appearance of maxprob
    return CkNN

#######################################################################################
################################ Decision Tree ########################################
#######################################################################################
def GetProb(C):
    # Prior a sample of belonging to class C
    Cm = []
    pm = []
    for ci in C: # for all samples
        if ci not in Cm:
            # Gather unseen class
            Cm.append(ci)
            pm.append(0)
        # Add to sum for that class
        Cmi = Cm.index(ci)
        pm[Cmi] += 1
    return Cm, np.asarray(pm)/len(C)

def NodeImpurity(pm, imptype):
    # Calculate impurity of node 
    e = 0.0
    if imptype == 'entropy':
        for pi in pm:
            e += -pi*np.log2(pi)
    elif imptype == 'gini':
        for pi in pm:
            e = 0.0
    elif imptype == 'misclassification error':
        for pi in pm:
            e = 0.0
    return e

def SplitImpurity(C, n):
    # Get the combined entropy of all resulting
    # branches after splitting a node
    e = 0.0
    for j in n:
        ej = 0.0
        Cj = [C[i] for i in j] # gather classes belonging to j
        [Cmj, pmj] = GetProb(Cj) # get priors
        for pmji in pmj:
            ej += -pmji*np.log2(pmji)
        e += len(j)*ej/len(C) 
    return e

def SplitIndex(X):
    # Splitting a discrete valued array based on like instances
    n = []
    j = []
    for t, xt in enumerate(X):
        if xt not in j: # gather new value for instance
            j.append(xt)
            n.append([])
        nidx = j.index(xt) # add to like valued group
        n[nidx].append(t)
    return n # returns array of index arrays for each group

def SplitAttribute(X, dtype):
    # Best split determined at minimum impurity
    # for any split at anny attribute
    MinEnt = float('inf')
    [N,D] = X.shape
    for i in range(D)[:-1]: # for all attributes
        if dtype == 'discrete':
            n = SplitIndex(X[:,i])
            e = SplitImpurity(X[:,-1], n)
            if e < MinEnt: # minimize impurity
                MinEnt = e
                besti  = i
                bestn  = n
        elif dtype == 'numeric':
            sort = list(X[:,i].argsort()) # sort indices by increasing X
            for t in range(N)[1:]:
                n = [ sort[0:t] , sort[t:N] ] # split indices
                e = SplitImpurity(X[:,-1], n)
                if e < MinEnt:
                    MinEnt = e
                    besti  = i
                    bestn  = n
    return besti, bestn # returns best feature to split in, and best way to split

def AddTerminal(T, C):
    # Create leaf on tree
    T['type']  = 'terminal'
    T['class'] = C
    return T

def AddNode(T, index, dtype):
    # Create a junction on tree
    T['type']  = 'node'
    T['index'] = index # attribute to compare
    if dtype == 'discrete':
        T['values'] = [] # possible values of branches
    elif dtype == 'numeric':
        T['midpoint'] = [] # 2 branches around midpoint
        T[   'above'] = {} # creating branches 
        T[   'below'] = {}
    return T

def AddBranch(T, val, dtype):
    # Create new branch on tree based on indicator
    if dtype == 'discrete':
        T['values'].append(val)
        T[    val ] = {}
    elif dtype == 'numeric':
        if val not in T['midpoint']:
            T['midpoint'].append(val)
    return T

def GetValue(X, i, j, n, dtype):
    # Find value for branch indicator
    if dtype == 'discrete':
        val = X[j[0],i] # value of all rows in branch
        tooth = val
    elif dtype == 'numeric':
        val  = (np.max(X[n[0],i]) + np.min(X[n[1],i]))/2 # midpoint between branches
        key = ['below', 'above']
        tooth = key[ n.index(j) ]
    return val, tooth

def GenerateTree(X, T, imptype, entmax, level, depth, depthmax, dtype):
    # Train a dataset and output a nested dictionary holding structure of tree
    # also output the depth of the tree
    # Supports numeric and discrete data
    # Stop conditions include impurity max and depth max
    level = level + 1 # add a level to the tree
    if level > depth: # keep highest level as depth
        depth = level
    [Cm, pm] = GetProb(X[:,-1]) # get priors for instances in class
    # If we meet an impurity threshold or reach user defined max depth, add leaf
    if NodeImpurity(pm, imptype) < entmax or depth == depthmax:
        T = AddTerminal(T, Cm[ list(pm).index(np.max(pm)) ])
        return [depth, T]
    # Otherwise define position as node and keep adding branches
    else:
        [i, n] = SplitAttribute(X, dtype) # minimum entropy split
        T = AddNode(T, i, dtype) # save index of split in node definition
        for j in n: # for all groups in split
            [val, tooth] = GetValue(X, i, j, n, dtype) # get comparison value and branch indicator
            T = AddBranch(T, val, dtype) # add branch to tree
            Xj = X[j,:] # new X for that branch
            # Continue algorithm down new branch
            depth = GenerateTree(Xj, T[tooth], imptype, entmax, level, depth, depthmax, dtype)[0]
    return [depth, T]

def TreeClassify(X, T, dtype):
        if T['type'] == 'terminal': # if at a leaf
            C = T['class']
            return C
        elif dtype == 'discrete':
            i = T['index'] # index to compare value
            dmin = float('inf')
            for val in T['values']:
                dist = abs(X[i] - val) # find nearest to value L1 norm
                if dist < dmin:
                    dmin   = dist
                    branch = val
            C = TreeClassify(X, T[branch], dtype) # go down closest branch
            return C
        elif dtype == 'numeric':
            i  = T['index']
            mp = T['midpoint']
            if X[i] <= mp: # see if above or below midpoint
                branch = 'below'
            else:
                branch = 'above'
            C = TreeClassify(X, T[branch], dtype) # go down branch
            return C
#######################################################################################
################################## Performance ########################################
#######################################################################################
def perform(Ck, Ct): # CkNN, Ctest
    TF = np.zeros((4)) # TP, TN, FP, FN
    N  = Ck.shape[0]
    for n in range(N):
        if Ck[n] == Ct[n]:   # if true
            if Ck[n] == 4:   # if pos
                TF[0] += 1
            elif Ck[n] == 2: # if neg
                TF[1] += 1

        elif Ck[n] != Ct[n]: # if false
            if Ck[n] == 4:   # if pos
                TF[2] += 1
            elif Ck[n] == 2: # if neg
                TF[3] += 1
    return TF

def metrics(TF):
    # TF is [TP, TN, FP, FN]
    [TP, TN, FP, FN] = TF
    ACC = (TP + TN)/(sum(TF)) # accuracy
    TPR = TP/(TP + FN) # true positive rate, recall, or sensitivity 
    PPV = TP/(TP + FP) # positive predictive value or precision
    TNR = TN/(TN + FP) # true negative rate or specificity
    F1S = 2*PPV*TPR/(PPV + TPR)
    return ACC, TPR, PPV, TNR, F1S
#######################################################################################
#######################################################################################
#######################################################################################
