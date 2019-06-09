#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 01:02:14 2019

@author: chensijin
"""

import tensorflow as tf 
sess = tf.Session() 
a = tf.constant(1) 
b = tf.constant(2) 
print(sess.run(a+b))
