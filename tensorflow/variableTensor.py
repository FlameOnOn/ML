# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:17:28 2018

@author: sijinc
"""

import tensorflow as tf

if __name__ == "__main__":
    state = tf.Variable(0,name='counter')
    print(state.name)
    
    one = tf.constant(1)
    
    new_value = tf.add(state ,one)
    update=tf.assign(state,new_value)
    
    init = tf.global_variables_initializer()   # must have this if some variables are defined. and it is must be ran before session run others.
    
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(3):
            sess.run(update)
            print(sess.run(state))
