# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:02:07 2017

@author: yangsu
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#download data sets
#take about 30 minuites to get downloaded 

import tensorflow as tf

x = tf.placeholder(tf.float32,[None, 784])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# set up variables

y = tf.nn.softmax(tf.matmul(x,w) + b)

y_= tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
#cross_entropy

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
#minimize difference between analized image and trainning image
#done traning

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x:mnist.test.images,y_:mnist.test.labels }))
#testing 
#print accuracy of test image

