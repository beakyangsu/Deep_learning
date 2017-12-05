# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:42:05 2017

@author: yangsu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:27:24 2017

@author: yangsu
"""

import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

w = tf.Variable(-3.0)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = x * w

cost = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range(100) :  
    print(step, sess.run(w))
    sess.run(train)