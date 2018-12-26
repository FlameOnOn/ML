#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 21:47:25 2018

@author: chensijin
"""

import numpy as np
from sklearn import svm

if __name__ == "__main__":
    N=50
    np.random.seed(0)
    x = np.sort(np.random.uniform(0,6,N))
    y = 2*np.sin(x) + 0.1*np.random.randn(N)
    x = x.reshape(-1, 1)
    
    
    svr_rbf = svm.SVR(kernel='rbf', gamma=0.2, C=100)
    svr_rbf.fit(x, y)
    print("SVR, RBF,train:", svr_rbf.score(x, y))
    
    svr_linear = svm.SVR(kernel='linear', C=100)
    svr_linear.fit(x, y)
    print("SVR, linear,train:", svr_linear.score(x, y))
    
    svr_poly = svm.SVR(kernel='poly', degree=3, C=100)
    svr_poly.fit(x, y)
    print("SVR, poly,train:", svr_poly.score(x, y))
    
    x_test = np.linspace(x.min(), 1.1*x.max(), 100).reshape(-1,1)
    y_predict_rbf = svr_rbf.predict(x_test)
    y_predict_linear = svr_linear.predict(x_test)
    y_predict_poly = svr_poly.predict(x_test)
    
    
    
    