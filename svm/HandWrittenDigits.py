#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 21:21:06 2018

@author: chensijin
"""

import numpy as np
from sklearn import svm

def show_accuracy(a,b):
    acc= a.ravel() == b.ravel()
    print(float(acc.sum())/a.size)

if __name__=="__main__":
    path_train = '/Users/chensijin/Documents/git/ML/optdigits.tra'
    data = np.loadtxt(path_train, dtype=float, delimiter=',')
    x, y = np.split(data, (-1,), axis=1)
    y = y .ravel().astype(np.int)
    
    path_test = '/Users/chensijin/Documents/git/ML/optdigits.tes'
    data = np.loadtxt(path_test, dtype=float, delimiter=',')
    x_test, y_test = np.split(data, (-1,),axis=1)
    y_test = y_test.ravel().astype(np.int)
    
    clf = svm.SVC(C=1, kernel='rbf', gamma=0.001)
    clf.fit(x, y)
    
    y_predict = clf.predict(x_test)
    print(clf.score(x_test, y_test))
    show_accuracy(y_test, y_predict)
    
    
    
    