# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 21:54:30 2017

@author: yangsu
"""

import tensorflow as tf

x=[[0,1,2], [2,1,0]]

s=tf.Session()
#initialize to one,shape of x
t = tf.ones_like(x)
print(s.run(t))

#initialize to zero, shape of x
t = tf.zeros_like(x)
print("\n",s.run(t),"\n")


for x,y in zip([1,2,3], [4,5,6]):
    print(x,y)
    
print("\n")    
for x,y,z in zip([1,2,3], [4,5,6], [7,8,9]):
    print(x,y,z)