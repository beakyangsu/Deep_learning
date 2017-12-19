# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 21:11:37 2017

@author: yangsu
"""

import tensorflow as tf

sess = tf.Session()

#mean of integer type
mean = tf.reduce_mean([1,2], axis=0)
print("1:",sess.run(mean)) # 1

#mean all value
x = [[1.,2.], [3.,4.]]
mean = tf.reduce_mean(x)
print("2:",sess.run(mean)) # 2.5

#axixs=0 means row axis
mean = tf.reduce_mean(x, axis=0)
print("3:",sess.run(mean)) # [2,3]

#axixs=1 means col axis
mean = tf.reduce_mean(x, axis=1)
print("4:",sess.run(mean)) # [1.5,3.5]

#axixs=-1 calculate first one inside axis from last axis
mean = tf.reduce_mean(x, axis=-1)
print("5:",sess.run(mean)) # [1.5,3.5]


mean = tf.reduce_mean(tf.reduce_sum(x, axis=-1))
print("6:",sess.run(mean)) # 5.0


x = [[0,1,2], [2,1,0]]

# position of max value
max = tf.argmax(x, axis=0)
print("7:",sess.run(max)) # [1,0,0]

max = tf.argmax(x, axis=1)
print("7:",sess.run(max)) # [2,0] 

max = tf.argmax(x, axis=-1)
print("7:",sess.run(max)) # [2,0]


