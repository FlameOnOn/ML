#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 23:07:47 2018

@author: chensijin
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
    y_ori = np.array([1,1,1,1,0,0])
    y_predict = np.array([1,0,1,1,1,1])
    
    print("Accuracy", accuracy_score(y_ori, y_predict))
    
    #tp / (tp + fp)
    print("Precission", precision_score(y_ori, y_predict))
    
    #tp / (tp + fn)
    print("Recall", recall_score(y_ori, y_predict))
    
    #F1 = 2 * (precision * recall) / (precision + recall)
    print("f1 score", f1_score(y_ori, y_predict))
    
    #fbeta = (1+beta**2)*precision*recall / (beta**2 * precision + recall)
    for beta in np.logspace(-3, 3, num=7, base=10):
        fbeta = fbeta_score(y_ori, y_predict, beta=beta)
        print("f", beta, "score", fbeta)
    
    print(precision_recall_fscore_support(y_ori, y_predict, beta=1))
    
    
