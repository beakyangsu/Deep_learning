# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:04:15 2017

@author: yangsu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:42:46 2017

@author: yangsu
"""

import tensorflow as tf


tf.set_random_seed(777)

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

x_col = len(x_data[0])

print(x_data.shape, x_data, len(x_data), len(x_data[0]))
print(y_data.shape, y_data)


#binary classification

x = tf.placeholder(tf.float32, shape=[None, x_col])
y = tf.placeholder(tf.float32, shape=[None, 1])
w = tf.Variable(tf.random_normal([x_col,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


hypothesis = tf.sigmoid(tf.matmul(x , w)+ b)

cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)



#True if hypothesis >0.5 else false, use tf.cast for true, false being 1,0
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict = {x : x_data, y: y_data})
        if step % 20 == 0:
           print(step, cost_val)
            
     # y_data for run accuracy       
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x : x_data, y: y_data})
    print("\nhypothesis : \n", h, "\npredicted : \n",p, "\naccuracy : ", a * 100 , "%" )
            
            
