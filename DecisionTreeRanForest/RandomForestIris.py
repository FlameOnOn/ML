#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 23:40:54 2018

@author: chensijin
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing 
if __name__ == "__main__":
    path = "/Users/chensijin/Documents/git/ML/iris.data"
    
    data = pd.read_csv(path)
    x = data.values[:, :-1]
    y = data.values[:, -1]
    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    y = le.transform(y)
    
    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    
    for i, pair in enumerate(feature_pairs):
        x_new = x[:, pair]
        clf = RandomForestClassifier(n_estimators=10, criterion="entropy", max_depth=4)
        clf = clf.fit(x_new, y)
        print(clf.score(x_new, y))
        y_predict = clf.predict(x_new)
        c = np.count_nonzero(y_predict == y)
        print(c / len(y))
        result = (y_predict == y)
        acc = np.mean(result)
        print(acc)