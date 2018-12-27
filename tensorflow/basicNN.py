# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:30:46 2018

@author: sijinc
"""

import tensorflow as tf
import numpy as np
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+ 0.1) 
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

if __name__ == "__main__":
    x_data = np.linspace(-1,1,300)[:,np.newaxis]
    noise = np.random.normal(0,0.05,x_data.shape)  ## zheng tai
    y_data = np.square(x_data)-0.5 + noise
    
    xs = tf.placeholder(tf.float32,[None,1])
    ys = tf.placeholder(tf.float32,[None,1])
    l1 = add_layer(xs, 1, 10,activation_function=tf.nn.relu)  #输入是x，属性只有1个，因为我们想定义隐藏层有10个神经元，所以输出是10，激活函数用relu
    predition = add_layer(l1, 10,1,activation_function=None)
    
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),reduction_indices=[1]))
    
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    for i in range(1000):
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
        if i%50==0:
            print(sess.run(loss, feed_dict={xs:x_data,ys:y_data}))
