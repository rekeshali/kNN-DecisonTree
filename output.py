#!/bin/usr/python
import numpy as np

def kNNout(klist, Lnormlist, iters, TF, M):
    # TF = [ TP, TN, FP, FN ]
    # M  = [ ACC, TPR, PPV, TNR, F1S]
    # TF and M have sizes [ len(Lnormlist), 4 for TF/5 for M, len(klist) ]
    import pandas as pd
    fname = 'kNN' + str(iters) + '.xlsx'
    out = pd.ExcelWriter(fname, engine='xlsxwriter')
    Mname = ['Accuracy','True Positive Rate','Positive Predictive Value','True Negative Rate','F1 Score']
    keyC = ['benign','malignant']
    keyM = ['Ln,k'] + klist

    for ki,k in enumerate(klist): # print confusion matrix for each Lnorm,k pair on same sheet
        for li, Lnorm in enumerate(Lnormlist):
            dbC = {}
            text = "Ln=" + str(Lnorm) + ', k=' + str(k)
            dbC[text]    = keyC
            dbC[keyC[0]] = [ TF[li,1,ki], TF[li,3,ki] ]
            dbC[keyC[1]] = [ TF[li,2,ki], TF[li,0,ki] ]
            DF = pd.DataFrame(dbC)
            DF.to_excel(out, sheet_name="Confusion", index=False, startrow=li*4, startcol=ki*4)

    for i in range(5): # print metrics of Lnorm,k on separate sheets
        dbM = {}
        for t,tooth in enumerate(keyM):
            if t == 0:
                dbM[tooth] = Lnormlist
            else:
                dbM[tooth] = M[:,i,t-1]
        DF = pd.DataFrame(dbM)
        DF.to_excel(out, sheet_name=Mname[i], index=False)

    out.save()

def TreeOut():
    return 0
