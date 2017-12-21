# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:44:33 2017

@author: yangsu
"""
import tensorflow as tf
import numpy as np

# 1-dimention
t = np.array([0.,1.,2.,3.,4.,5.,6.])

print(t.ndim)
print(t.shape)
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

#2-dimension
t = np.array([[1.,2.,3.], [4.,5.,6.], [7.,8.,9.], [10.,11.,12.]])
print(t.ndim)
print(t.shape)

sess=tf.Session()
#tenser
t = tf.constant([1,2,3,4])
print(sess.run(tf.shape(t)))
#(array[4])
t = tf.constant([[1,2], [3,4]])
print(sess.run(tf.shape(t)))
#(array[2,2])
t = tf.constant([[[[1,2,3,4], [5,6,7,8], [9,10,11,12]], [[13,14,15,16], [17,18,19,20], [21,22,23,24]]]])
print(sess.run(tf.shape(t)))
#(array[1,2,3,4])
