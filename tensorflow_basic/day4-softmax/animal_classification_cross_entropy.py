# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:08:38 2017

@author: yangsu
"""

import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]
nb_classes = 7

x = tf.placeholder(np.float32, shape=[None, 16])
y = tf.placeholder(np.int32, shape=[None, 1]) #0~6 (?,1)
y_one_hot = tf.one_hot(y, nb_classes) #one_hot (?, 1,7)
y_one_hot = tf.reshape(y_one_hot, [-1, nb_classes]) #shape == (?,7)



w = tf.Variable(tf.random_normal([16,nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

logits = tf.matmul(x,w)+b
h = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


prediction = tf.argmax(h,1)
correct_prediction = tf.equal(prediction, tf.argmax(y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(optimizer, feed_dict={x:x_data, y:y_data})
        if step % 20 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict = {x:x_data, y: y_data})
            print("step,{:5}\tLoss: {:3}\tAcc: {:.2%}".format(step,loss,acc))
            
    pred = sess.run(prediction, feed_dict={x:x_data})
    for p, y in zip(pred, y_data.flatten()) :
        print("[{}] prediction : {} TRUE y : {}".format(p==int(y), p, int(y)))