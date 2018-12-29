# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 17:20:13 2018

@author: sijinc
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def show_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def con2d(x, W):  #x是输入的值，图片的值什么的，W是weight
    #stride: [1,x_movement,y_movement,1]
    #padding有两种，valid，same，valid抽取出来的比原图片小，same抽取出来的和原图片一样
    return tf.nn.conv2d(x, W, strides=[1,1,1,1],padding='SAME')  #2维的卷积神经网络.步长，两边的1是固定的，中间的两个1一个是水平方向步长为1，一个是竖直方向步长为1

def max_pool_2x2(x):
    #stride: [1,x_movement,y_movement,1]
    #在pooling阶段把图片压缩了
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')  #average pooling and max pooling


if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1,28,28,1]) #-1 就是不管它，会自己算的，28 * 28就是图片的像素矩阵，1是channel，通道，这个是黑白的，所以channel是1，要是有RGB的话，就是3
    #print(x_image.shape)  # [n_samples, 28,28,1]
    
    #start to define conv layer
    #conv1 layer
    W_conv1 = weight_variable([5,5,1,32]) #卷积核patch的大小是5x5，因为黑白图片channel是1所以输入是1，输出是32个featuremap
    b_conv1 = bias_variable([32])
    h_convl = tf.nn.relu(con2d(x_image, W_conv1) + b_conv1) #output size 28x28x32
    h_pool1 = max_pool_2x2(h_convl) #pooling output size 14x14x32 ,因为pooling的步长是2
    #conv2 layer
    W_conv2 = weight_variable([5,5,32,64]) #卷积核patch的大小是5x5，因为黑白图片channel是1所以输入是1.因为有32个feature map, 输入图像的厚度（channel）就变成了32
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(con2d(h_pool1, W_conv2) + b_conv2) #output size 14x14x64
    h_pool2 = max_pool_2x2(h_conv2) #pooling output size 7x7x64 ,因为pooling的步长是2
    #function1 layer
    W_fc1 = weight_variable([7*7*64, 1024]) #第二个卷积层展平了的输出大小: 7x7x64， 后面的输出size我们继续扩大，定为1024
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    #functin2 layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    prediction=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    #loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
    
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys:batch_ys, keep_prob: 0.5})
        if i%50 == 0:
            print(show_accuracy(mnist.test.images, mnist.test.labels))
