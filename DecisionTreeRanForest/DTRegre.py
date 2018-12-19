#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 23:57:30 2018

@author: chensijin
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor

if __name__ == '__main__':
    N = 100
    x = np.random.rand(N) * 6 - 3   #rand jun yun
    x.sort()
    y = np.sin(x) + np.random.randn(N) * 0.03    #randn zheng tai
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    print(x.shape)
    print(y.shape)
    reg = DecisionTreeRegressor(criterion='mse', max_depth=3)
    dtr = reg.fit(x, y)
    x_test = np.linspace(-3, 3, 50).reshape(-1,1)
    y_predict = dtr.predict(x_test).reshape(-1,1)
    
    print(y_predict)
    print(dtr.score(x, y))
    
    depth = [2, 4, 6, 8, 10]
    reg = [DecisionTreeRegressor(criterion='mse', max_depth=depth[0]),
           DecisionTreeRegressor(criterion='mse', max_depth=depth[1]),
           DecisionTreeRegressor(criterion='mse', max_depth=depth[2]),
           DecisionTreeRegressor(criterion='mse', max_depth=depth[3]),
           DecisionTreeRegressor(criterion='mse', max_depth=depth[4])]
    
    for i, r in enumerate(reg):
        dt = r.fit(x, y)
        y_predict_new = dt.predict(x_test)
        print(y_predict)
        print(dt.score(x, y))