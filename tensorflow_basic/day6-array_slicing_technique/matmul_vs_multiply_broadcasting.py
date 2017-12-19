# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 21:01:28 2017

@author: yangsu
"""

import tensorflow as tf

mat1 = tf.constant([[1.,2.], [3.,4.]])
mat2 = tf.constant([[1.], [2.]])
print("Mat 1 shape : ", mat1.shape)
print("Mat 2 shape : ", mat2.shape)

sess=tf.Session()
print("matmul :\n", sess.run(tf.matmul(mat1,mat2)))
print("multipy : \n", sess.run(mat1*mat2))


#add :same shape 
mat1 = tf.constant([[3.], [3.]])
mat2 = tf.constant([[2.], [2.]])
print("add : \n", sess.run(mat1+mat2))


#add :different shape == broadcasting
mat1 = tf.constant([[1., 2.]])
mat2 = tf.constant(3.)

print("add : \n", sess.run(mat1+mat2))

mat1 = tf.constant([[1., 2.]])
mat2 = tf.constant([[3., 4.]])

print("add : \n", sess.run(mat1+mat2))

mat1 = tf.constant([[1., 2.]])
mat2 = tf.constant([[3.], [4.]])
print("add : \n", sess.run(mat1+mat2))