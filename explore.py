#!/usr/bin/ python
import numpy as np
# macro to check for data format
def is_num(string): # returns true if number, false otherwise
    try:
        float(string)
        return True
    except ValueError:
        return False# open, get, close data files

# begin data exploration
import csv
dbraw = {}
key = ['ID','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
for tooth in key:
    dbraw[tooth] = []
badidx = []
filename = 'data/breast-cancer-wisconsin.data'
with open(filename) as csv_file: # create dict of all attributes
    csv_reader = csv.reader(csv_file)
    for t, row in enumerate(csv_reader):
        for j,item in enumerate(row):
            dbraw[key[j]].append(item)
            if (is_num(item)) == False:
                if t not in badidx:
                    badidx.append(t)

for tooth in key:
    for idx in badidx[::-1]:
        dbraw[tooth].pop(idx)

for tooth in key:
    for idx,item in enumerate(dbraw[tooth]):
        dbraw[tooth][idx] = float(item)

# X = np.zeros( ( len(dbraw[key[0]]) , len(key) - 1) )
# i = 0
# for tooth in key:
#     if tooth == key[0]:
#         continue
#     X[:, i] = dbraw[tooth]
#     i = i + 1

