# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 21:51:30 2017

@author: yangsu
"""

import tensorflow as tf

x=[1,4]
y=[2,5]
z=[3,6]

s=tf.Session()
st=tf.stack([x,y,z])
print(s.run(st))

#[[1 4]
#[2 5]
#[3 6]]

st=tf.stack([x,y,z], axis=1)
print(s.run(st))
#[[1 2 3]
#[4 5 6]]