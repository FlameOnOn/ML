# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 09:57:50 2018

@author: sijinc
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def show_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1)) #看看输出概率最大的那个位置是不是和真实值为1的按个位置相同。相同就预测对了，不同就预测错了
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys:v_ys})
    return result

if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    xs = tf.placeholder(tf.float32, [None, 784]) #28 * 28
    ys = tf.placeholder(tf.float32, [None, 10])
    
    prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
    
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys:batch_ys})
        if i%50 == 0:
            print(show_accuracy(mnist.test.images, mnist.test.labels))
