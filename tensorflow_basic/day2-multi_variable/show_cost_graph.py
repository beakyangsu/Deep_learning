# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:20:01 2017

@author: yangsu
"""

import tensorflow as tf
import matplotlib.pyplot as plt

x = [1,2,3]
y = [1,2,3]

w = tf.placeholder(tf.float32)

h = x*w

cost = tf.reduce_mean(tf.square(h-y))
sess = tf.Session()

sess.run(tf.global_variables_initializer())

w_val = []
cost_val = []

for i in range(-30, 50) :
    feed_w = i*0.1
    curr_cost, curr_w = sess.run([cost, w], feed_dict={w : feed_w})
    w_val.append(curr_w)
    cost_val.append(curr_cost)
    
    
plt.plot(w_val, cost_val)
plt.show( )
    