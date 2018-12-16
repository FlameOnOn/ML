#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 22:50:35 2018

@author: chensijin
"""

def iris_type(s):
    dict = {'a': 1, 'b': 2, 'b': 3}
    return dict[s]

if __name__ == '__main__':
    #dict = {'a': 1, 'b': 2, 'b': 3}
    print(iris_type('a'))