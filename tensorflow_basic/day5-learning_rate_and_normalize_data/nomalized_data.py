# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:48:38 2017

@author: yangsu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:20:49 2017

@author: yangsu
"""

import tensorflow as tf
import numpy as np

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

tf.set_random_seed(777)  # for reproducibility

# fluctuated data value
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])


#normalize
xy_n = MinMaxScaler(xy)
print("\nnormalized data: \n", xy_n)

xn_data = xy_n[:, 0:-1]
yn_data = xy_n[:, [-1]]

x = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,1])
w = tf.Variable(tf.random_normal([4,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

h = tf.matmul(x,w)+b
cost = tf.reduce_mean(tf.square(h-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())
    

for step in range(101):
    cost_val, hy_val, _ = sess.run(
        [cost, h, train], feed_dict={x: xn_data, y: yn_data})
    print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)



