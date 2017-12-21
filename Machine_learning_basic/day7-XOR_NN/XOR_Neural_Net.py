# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 22:54:08 2017

@author: yangsu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 22:35:36 2017

@author: yangsu
"""
import tensorflow as tf
import numpy as np

x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([[0], [1], [1], [0]])

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


w1 = tf.Variable(tf.random_normal([2,2]), name ='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(x,w1)+b1)

w2 = tf.Variable(tf.random_normal([2,1]), name ='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1,w2)+b2)
#Neural Net


cost = -tf.reduce_mean(y*tf.log(layer2)+(1-y)*tf.log(1-layer2))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

predict = tf.cast(layer2 > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y, predict), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(train, feed_dict = {x:x_data, y:y_data})
        
        if step % 20 == 0:
           print('step: %d , cost : %.2f' % (step, sess.run(cost, feed_dict ={x:x_data, y:y_data})))
            
            
            
    print('accuracy : %.2f' %sess.run(accuracy, feed_dict = { x:x_data,y:y_data}))