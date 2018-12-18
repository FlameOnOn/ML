#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 21:55:35 2018

@author: chensijin
"""

import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    path = "/Users/chensijin/Documents/git/ML/iris.data"
    
    data = pd.read_csv(path)
    x = data.values[:, :-1]
    y = data.values[:, -1]
    #print(x)
    #print(y)
    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    #print(le.classes_)
    y = le.transform(y)
    #print(x.shape)
    #print(y, y.shape)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)
    model = Pipeline([('ss', StandardScaler()), ('DT', DecisionTreeClassifier(criterion='gini', max_depth=5))])
    model = model.fit(x_train, y_train)
    y_test_hat = model.predict(x_test)
    print(y_test_hat)
    print(y_test)
    print(model.score(x_test,y_test))
    result = (y_test_hat == y_test)
    acc = np.mean(result)
    print(acc)
    
    depth = np.arange(1, 15)
    print(depth)
    
    for d in depth:
        ss = StandardScaler()
        ss.fit(x_train)
        #dtc = Pipeline([('ss', StandardScaler()), ('dt', DecisionTreeClassifier(criterion='entropy', max_depth=d))])
        dtc = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        dtc = dtc.fit(x_train, y_train)
        print(d, dtc.score(x_test, y_test))
    
    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    for i, pair in enumerate(feature_pairs):
        x_train_new = x_train[:, pair]
        x_test_new = x_test[:, pair]
        #ss = StandardScaler
        #ss.fit(x_train_new)
        new_dtc = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=3, min_samples_split=10)
        new_dtc.fit(x_train_new,y_train)
        
        print(new_dtc.score(x_test_new, y_test))
        
    