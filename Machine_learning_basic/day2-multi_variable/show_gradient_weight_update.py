# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:27:24 2017

@author: yangsu
"""

import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

w = tf.Variable(tf.random_normal([1]), name='weight')
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = x * w

cost = tf.reduce_sum(tf.square(hypothesis-y))
 

learning_rate = 0.1
gradient = tf.reduce_mean(w*x-y)*x
descent = w - learning_rate *  gradient

#tensor flow는 =으로 값 할당 못함 

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range(21) :
    
    sess.run(descent, feed_dict = {x : x_data, y: y_data})
    print(step, sess.run(cost, feed_dict={x : x_data, y : y_data}), sess.run(w))
   