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