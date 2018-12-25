# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 09:48:41 2018

@author: sijinc
"""

import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import csv

def show_accrracy(a, b):
    acc = a.ravel() == b.ravel()
    acc_rate = float(acc.sum())/a.size
    return acc_rate

def load_data(file_name, is_train):
    data = pd.read_csv(file_name)
    print(data.describe())
    
    data['Sex'] = data['Sex'].map({'female':0, 'male':1}).astype(int)
    if len(data.Fare[data.Fare.isnull()]) > 0:
        fare = np.zeros(3)
        for f in range(0, 3):
            fare[f] = data[data.Pclass == f+1]['Fare'].dropna().median()
        for f in range(0, 3):
            data.loc[(data.Fare.isnull()) & (data.Pclass == f+1), 'Fare'] = fare[f]
    
    #mean_age = data['Age'].dropna().mean()
    #data.loc[(data.Age.isnull()), 'Age'] = mean_age
    if is_train:
        #use RD to predict ages
        data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]
        age_null = data_for_age.loc[(data.Age.isnull())]
        x = age_exist.values[:,1:]
        y = age_exist.values[:,0]
        
        rfr = RandomForestRegressor(n_estimators=1000)
        rfr.fit(x, y)
        
        age_predict = rfr.predict(age_null.values[:, 1:])
        data.loc[(data.Age.isnull()), 'Age'] = age_predict
    else:
        data_for_age = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]
        age_null = data_for_age.loc[(data.Age.isnull())]
        
        x = age_exist.values[:,1:]
        y = age_exist.values[:,0]
        
        rfr = RandomForestRegressor(n_estimators=1000)
        rfr.fit(x, y)
        age_predict = rfr.predict(age_null.values[:,1:])
        data.loc[(data.Age.isnull()), 'Age'] = age_predict
    
    data.loc[(data.Embarked.isnull()), 'Embarked'] = 'S'
    embarked_data = pd.get_dummies(data.Embarked)
    embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
    data = pd.concat([data, embarked_data], axis=1)
    print(data.describe())
    
    x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    y = None
    if 'Survived' in data:
        y = data['Survived']
    
    x = np.array(x)
    y = np.array(y)
    
    if is_train:
        return x, y
    return x, data['PassengerId']

if __name__ == "__main__":
    path_train = 'D:\\mldata\\XGBoost\\Titanic.train.csv'
    x, y = load_data(path_train, True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    lr_rate = show_accrracy(y_predict, y_test)
    print("Logistic: ", lr_rate)
    
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(x_train, y_train)
    y_predict = rfc.predict(x_test)
    rfc_rate = show_accrracy(y_predict, y_test)
    print("RF_100: ", rfc_rate)
    
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
    bst = xgb.train(param, data_train, num_boost_round=100, evals=watch_list)
    y_predict = bst.predict(data_test)
    y_predict[y_predict>0.5]=1
    y_predict[~(y_predict>0.5)]=0
    xgb_rate = show_accrracy(y_predict,y_test)
    print("XGBoost: " , xgb_rate)
