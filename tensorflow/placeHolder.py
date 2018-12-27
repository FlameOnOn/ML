# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:26:14 2018

@author: sijinc
"""

import tensorflow as tf

if __name__ == "__main__":
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    
    output = tf.multiply(input1,input2)
    
    with tf.Session() as sess:
        print(sess.run(output, feed_dict={input1:[7],input2:[2]}))
