# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 21:33:58 2017

@author: yangsu
"""

import tensorflow as tf
import numpy as np

t = np.array([[[0,1,2], [3,4,5]], [[6,7,8], [9,10,11]]])
print(t.shape)

sess=tf.Session()

#reshape, -1 == None
re = tf.reshape(t, shape=[-1,3])
print("2D:\n",sess.run(re))

#reshape, -1 == None
re = tf.reshape(t, shape=[-1,1,3])
print("3D:\n",sess.run(re))

#reshape : squeeze
sq = tf.squeeze([[0], [1], [2]])
print("squeeze:\n",sess.run(sq))

#reahpe : expand
ex = tf.expand_dims([0,1,2], 1)
print("expand:\n",sess.run(ex))