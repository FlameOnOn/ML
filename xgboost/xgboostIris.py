# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 16:11:08 2018

@author: sijinc
"""

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing

if __name__ == "__main__":
    path = 'D:\\mldata\\XGBoost\\iris.data'
    data = pd.read_csv(path)
    x = data.values[:,:-1]
    y = data.values[:,-1]
    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    y = le.transform(y)
    print(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.75, test_size=0.25)
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth':5, 'eta':1, 'silent':1, 'objective':'multi:softmax', 'num_class':3}
    bst = xgb.train(param, data_train, num_boost_round=6, evals=watch_list)
    
    y_predict = bst.predict(data_test)
    result = y_test = y_predict
    print(float(np.sum(result))/len(y_predict))
    print(np.average(result))
    
    #[0]     eval-merror:0.026316    train-merror:0.018018
    #[1]     eval-merror:0.026316    train-merror:0.018018
    #[2]     eval-merror:0.026316    train-merror:0
    #[3]     eval-merror:0.026316    train-merror:0
    #[4]     eval-merror:0.026316    train-merror:0
    #[5]     eval-merror:0.026316    train-merror:0
