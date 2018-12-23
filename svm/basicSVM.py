#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 00:11:04 2018

@author: chensijin
"""
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing

if __name__ == "__main__":
    path = "/Users/chensijin/Documents/git/ML/iris.data"
    data = pd.read_csv(path)
    x = data.values[:,:-1]
    y = data.values[:,-1]
    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    y = le.transform(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6, test_size=0.4)
    clfRBF = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clfRBF.fit(x_train, y_train)
    print(clfRBF.score(x_train, y_train)) #gamma=20, already overfitting
    print(clfRBF.score(x_test, y_test))
    clfL = svm.SVC(C=0.8, kernel='linear', decision_function_shape='ovr')
    clfL.fit(x_train, y_train)
    print(clfL.score(x_train, y_train))
    print(clfL.score(x_test, y_test))