#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 23:25:48 2018

@author: chensijin
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor

if __name__ == "__main__":
    N = 300
    x = np.random.rand(N) * 8 - 4
    x.sort()
    y1 = np.sin(x) + 3 + np.random.randn(N) * 0.1
    y2 = np.cos(0.3*x) + np.random.randn(N) * 0.01
    
    y = np.vstack((y1, y2))
    y = y.T
    
    print(y)
    x = x.reshape(-1, 1)
    
    #depth = 3
    depth = 10
    reg = DecisionTreeRegressor(criterion="mse", max_depth=depth)
    reg = reg.fit(x , y)
    
    x_test = np.linspace(-4, 4, num = 1000).reshape(-1, 1)
    y_test = reg.predict(x_test)
    
    print(x_test)
    print(y_test)
    
    print(reg.score(x , y))

