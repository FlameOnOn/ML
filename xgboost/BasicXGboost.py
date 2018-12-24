#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 23:14:26 2018

@author: chensijin
"""

import xgboost as xgb
import numpy as np
if __name__ == "__main__":
    path_train = 'D:\\mldata\\XGBoost\\agaricus_train.txt'
    path_test = 'D:\\mldata\\XGBoost\\agaricus_test.txt'
    data_train = xgb.DMatrix(path_train)
    data_test = xgb.DMatrix(path_test)
    
    param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
    param_logit_traw = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logitraw'}
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 3  #number of trees
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    bst_new = xgb.train(param_logit_traw, data_train, num_boost_round=n_round, evals=watchlist)
    y_predict = bst.predict(data_test)
    y = data_test.get_label()
    print(y_predict)
    print(y)
    error = sum(y != (y_predict > 0.5))
    error_rate = float(error)/len(y_predict)
    print(error_rate)
    
    y_predict_new = bst_new.predict(data_test)
    print(y_predict_new)
    error = sum(y != (y_predict_new > 0))
    error_rate = float(error)/len(y_predict)
    print(error_rate)

