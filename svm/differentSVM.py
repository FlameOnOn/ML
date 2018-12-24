#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 21:54:52 2018

@author: chensijin
"""

import numpy as np
from sklearn import svm

def show_accuracy(a, b):
    acc = a.ravel() == b.ravel()
    print(float(acc.sum())/a.size)

if __name__ == "__main__":
    path = '/Users/chensijin/Documents/git/ML/bipartition.txt'
    data = np.loadtxt(path, dtype=float, delimiter='\t')
    x, y = np.split(data, (2,), axis = 1)
    y[y == 0] = -1
    y = y.ravel()
    
    clfs = [svm.SVC(C=0.3, kernel='linear'),
           svm.SVC(C=10, kernel='linear'),
           svm.SVC(C=5, kernel='rbf', gamma=1),
           svm.SVC(C=5, kernel='rbf', gamma=4)]
    
    for i, clf in enumerate(clfs):
        clf.fit(x,y)
        y_predict = clf.predict(x)
        #print(clf.score(x,y))
        show_accuracy(y_predict, y)
        print(clf.n_support_)  #number of supporting vector.because there are two classes, so [36,37] means 36, and 37 supporting vectors on each side
        print(clf.dual_coef_)  #parms of supporting vector
        print(clf.support_)    # which of the samples are the supporting vector
        