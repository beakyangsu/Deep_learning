# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 21:44:17 2017

@author: yangsu
"""

import tensorflow as tf

s=tf.Session()
#depth == the number of most inside dimension
#auto expand 
#tf.one_hot
one=tf.one_hot([[0], [1], [2], [0]], depth=3)
print(s.run(one))

re=tf.reshape(one, shape=[-1,3]) 
print(s.run(re))

ca=tf.cast([1.8,2.2,3.3,4.9], tf.int32)
print(s.run(ca))

ca=tf.cast([True, False, 1==1, 0==1], tf.int32)
print(s.run(ca))