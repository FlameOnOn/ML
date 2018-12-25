# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 15:11:14 2018

@author: sijinc
"""

import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import csv

def f(x):
    return 0.5*np.exp(-(x+3) **2) + np.exp(-x**2) + + 0.5*np.exp(-(x-3) ** 2)

if __name__ == "__main__":
    np.random.seed(0)
    N = 200
    x = np.random.rand(N) * 10 - 5
    x = np.sort(x)
    y = f(x) + 0.05*np.random.randn(N)
    x.shape = -1,1
    
    ridge = RidgeCV(alphas=np.logspace(-3, 2, 10), fit_intercept=False)
    ridged = Pipeline([('poly', PolynomialFeatures(degree=8)), ('Ridge', ridge)])
    bagging_ridged = BaggingRegressor(ridged, n_estimators=100, max_samples=0.3)
    dtr = DecisionTreeRegressor(max_depth=5)
    
    regs=[('DTR', dtr),
          ('Ridge R(8 Degre)', ridged),
          ('Bagging Ridge(8-degree)', bagging_ridged),
          ('Bagging DTR', BaggingRegressor(dtr, n_estimators=100,max_samples=0.3))]
    
    for i, (name, reg) in enumerate(regs):
        reg.fit(x, y)
        print(name, reg.score(x, y))
