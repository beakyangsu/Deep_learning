# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 22:06:04 2017

@author: yangsu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:51:51 2017

@author: yangsu
"""

import tensorflow as tf

x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

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
    if step % 20 == 0 :
        print(step, "cost:" ,cost_val, "hy :", hy_val)
        
        
