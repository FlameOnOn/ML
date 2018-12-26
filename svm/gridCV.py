#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 22:04:51 2018

@author: chensijin
"""

from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np

if __name__ == "__main__":
    N=50
    np.random.seed(0)
    x = np.sort(np.random.uniform(0,6,N), axis = 0)
    y = 2*np.sin(x) + 0.1*np.random.randn(N)
    x = x.reshape(-1,1)
    model = svm.SVR(kernel='rbf')
    c_can = np.logspace(-2, 2, 10)
    gamma_can = np.logspace(-2,2,10)
    svr = GridSearchCV(model, param_grid={'C':c_can, 'gamma':gamma_can}, cv=5)
    svr.fit(x, y)
    
    print(svr.best_params_, svr.score(x, y))
    sv = svr.best_estimator_.support_
    print(sv)