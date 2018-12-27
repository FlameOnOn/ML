# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:04:31 2018

@author: sijinc
"""

import tensorflow as tf

if __name__ == "__main__":
    matrix1 = tf.constant([[3,3]])
    matrix2 = tf.constant([[2],[2]])
    
    product = tf.matmul(matrix1,matrix2)
     
    #method 1
    sess = tf.Session()
    result = sess.run(product)
    print(result)
    sess.close()
    
    #method 2
    with tf.Session() as sess2:
        result2 = sess2.run(product)
        print(result2)
