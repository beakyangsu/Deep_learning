# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 20:51:05 2017

@author: yangsu
"""

import tensorflow as tf

x_data = [[1,2,1], [1,3,2], [1,3,4], [1,5,5], [1,7,5], [1,2,5], [1,6,6], [1,7,7]]
y_data = [[0,0,1], [0,1,0], [0,0,1], [0,1,0], [0,1,0], [0,0,1], [1,0,0], [1, 0, 0]]

x_test = [[2,1,1], [3,1,2], [3,3,4]]
y_test = [[0,0,1], [0,0,1], [0,0,1]]


x = tf.placeholder("float", [None,3])
y = tf.placeholder("float", [None,3])
w = tf.Variable(tf.random_normal([3,3]))
b = tf.Variable(tf.random_normal([3]))


h = tf.nn.softmax(tf.matmul(x,w)+b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(h), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-10).minimize(cost)

prediction = tf.arg_max(h,1)
is_correct = tf.equal(prediction, tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    for step in range(201) :
        cost_val, w_val, _ = sess.run([cost, w, optimizer], feed_dict={x:x_data, y:y_data})
        print(step, cost_val, w_val)
        
    p_val, cor_val = sess.run([prediction,is_correct], feed_dict={x:x_test, y:y_test})
    acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test})
    print("prediction:",p_val, "\nis_correct :" , cor_val, "\naccuracy :", acc)
   
    