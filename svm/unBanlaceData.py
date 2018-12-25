#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 21:55:39 2018

@author: chensijin
"""

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import warnings

def show_accuray(a,b):
    acc=a.ravel() == b.ravel()
    print(float(acc.sum())/a.size)

def show_recall(y, y_predict):
    print(float(np.sum(y_predict[y == 1] == 1))/np.extract(y==1,y).size)
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    
    c1 = 990
    c2 = 10
    N = c1 + c2
    x_c1= 3*np.random.randn(c1, 2)
    x_c2=0.5 * np.random.randn(c2,2) + (4,4)
    
    x = np.vstack((x_c1,x_c2))
    print(x.shape)
    y = np.ones(N)
    y[:c1] = -1 
    
    clfs = [svm.SVC(C=1, kernel='linear'),
            svm.SVC(C=1, kernel='linear', class_weight={-1:1, 1:10}),
            svm.SVC(C=0.8, kernel='rbf', gamma=0.5, class_weight={-1:1, 1:2}),
            svm.SVC(C=0.8, kernel='rbf', gamma=0.5, class_weight={-1:1, 1:10})]
    
    for i, clf in enumerate(clfs):
        clf.fit(x, y)
        y_predict = clf.predict(x)
        print(i)
        print("acc", accuracy_score(y, y_predict))
        print("pre",precision_score(y, y_predict))
        print("pre",precision_score(y, y_predict,pos_label=1)) #by default pos_lable is 1, means the label we are calculating, can set to -1(the other class) to have a look
        print("recall",recall_score(y, y_predict, pos_label=1))
        print("f1",f1_score(y,y_predict,pos_label=1))
        