# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 14:46:32 2018

@author: sijinc
"""

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biase = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biase
    
    #dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    
    return outputs

def show_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob:0.5})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1)) #看看输出概率最大的那个位置是不是和真实值为1的按个位置相同。相同就预测对了，不同就预测错了
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys:v_ys})
    return result

if __name__ =="__main__":
    digits = load_digits()
    x = digits.data
    y = digits.target
    y = LabelBinarizer().fit_transform(y) # transfor to one-hot
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    keep_prob = tf.placeholder(tf.float32)
    xs = tf.placeholder(tf.float32, [None, 64])  #这里 none的意思就是不管它有多少行的意思
    ys = tf.placeholder(tf.float32, [None, 10])
    
    l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)  # 隐藏层
    prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)  #输出层
    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])) #cost function
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_train, ys: y_train, keep_prob:0.5})
        if i % 50 == 0:
            print(sess.run(cross_entropy, feed_dict={xs:x_train,ys:y_train,keep_prob:0.5}))
            print(show_accuracy(x_test, y_test))
