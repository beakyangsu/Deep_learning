# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 22:09:03 2017

@author: yangsu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 22:01:46 2017

@author: yangsu
"""

import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])


#tf.Variable은 train변수, 학습하면서 알아서 바뀜
w = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

h = x * w + b

cost = tf.reduce_mean(tf.square(h-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, w_val, b_val, _ = \
    sess.run([cost,w,b, train], feed_dict={x:[1,2,3,4,5], y:[2.1,3.1,4.1,5.1,6.1]})
    
    if step % 20 == 0:
        print(step, cost_val, w_val, b_val)
    # == sess.run(train), sess.run(cost), sess.run(w), sess.run(b))
    
# train done

#test model, sess.run(train)으로 w,b값을 찾고 방정식(h) 노드를 실행시킴
print(sess.run(h, feed_dict={x:[5]}))
print(sess.run(h, feed_dict={x:[2.5]}))
print(sess.run(h, feed_dict={x:[1.5,3.5]}))