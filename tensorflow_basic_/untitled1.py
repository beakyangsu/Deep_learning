# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 22:01:46 2017

@author: yangsu
"""

import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


#tf.Variable은 train변수, 학습하면서 알아서 바뀜
w = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

h = x * w + b

cost = tf.reduce_mean(tf.square(h-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, w_val, b_val, _ = \
    sess.run([cost,w,b, train], feed_dict={x:[1,2,3], y:[1,2,3]})
    if step % 20 == 0:
        print(step, cost_val, w_val, b_val)
    # == sess.run(train), sess.run(cost), sess.run(w), sess.run(b))