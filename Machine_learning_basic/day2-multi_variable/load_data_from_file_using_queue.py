# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:12:49 2017

@author: yangsu
"""

import tensorflow as tf

filename_queue = tf.train.string_input_producer(['test-score.csv'], shuffle = False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)


recode_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, recode_defaults=recode_defaults)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x,w) + b

cost = tf.reduece_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOprimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#start populating the filename queue
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord = coord)

for step in range(2001) :
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x : x_batch, y: y_batch})
    if step % 20 == 0:
        print(step, "cost : ", cost_val, "\nprediction:" ,hy_val)


coord.request_stop()
coord.join(threads)