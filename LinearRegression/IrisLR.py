#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:55:09 2018

@author: chensijin
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def iris_type(s):
    irisType = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    return irisType[s]

if __name__ == "__main__":
    path = "/Users/chensijin/Documents/git/ML/iris.data"
    
    #data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type}) 
    #print(data)
    
    
    data = pd.read_csv(path)
    x = data.values[:, :-1]
    y = data.values[:, -1]
    #print(x)
    #print(y)
    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    print(le.classes_)
    y = le.transform(y)
    print(x.shape)
    print(y, y.shape)
    
    # x = StandardScaler().fit_transform(x)
    # lr = LogisticRegression()
    # lr.fit(x, y.ravel())
    lr = Pipeline([('sc', StandardScaler()), ('lrs', LogisticRegression())])  #softmax
    lr.fit(x,y)
    print("hhhhhhhhhhhhhhhhhhhhhhhhhhh")
    y_predict = lr.predict(x)
    print(y_predict)
    result = y_predict==y
    print(result)
    accuracy = np.mean(result)
    print(accuracy)