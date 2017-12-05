# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 22:13:20 2017

@author: yangsu
"""

import numpy as np
import tensorflow as tf

tf.set_random_seed(777)

xy = np.loadtxt('test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)


x = tf.placeholder(tf.float32, shape=[None,3])
y = tf.placeholder(tf.float32, shape=[None,1])
#shape = [# of instance, # of variable]


w = tf.Variable(tf.random_normal([3, 1]), name = 'wieght1')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(hypothesis-y))
Optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = Optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001) :
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x : x_data, y : y_data})
    #if step % 20 == 0 :
       # print(step, "cost:" ,cost_val, "hy :", hy_val)
        
        
print("score will be :" , sess.run(hypothesis, feed_dict={x:[[100,70,101]]}))