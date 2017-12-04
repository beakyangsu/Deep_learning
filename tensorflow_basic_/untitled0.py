# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:51:44 2017

@author: yangsu
"""

import tensorflow as tf

x = [1,2,3]
y = [1,2,3]

#tf.Variable은 train변수, 학습하면서 알아서 바뀜
w = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

h = x * w + b

cost = tf.reduce_mean(tf.square(h-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#tf.Variable 초기화 필수
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(w), sess.run(b))