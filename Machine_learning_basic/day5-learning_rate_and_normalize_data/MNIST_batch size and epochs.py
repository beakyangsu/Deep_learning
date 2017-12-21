# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:56:51 2017

@author: yangsu
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#0~9 digits
nb_classes = 10

## of pixels in one image
x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, nb_classes])
w = tf.Variable(tf.random_normal([784,nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]))

h = tf.nn.softmax(tf.matmul(x,w)+b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(h), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(h,1), tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# number of train about full examples 
training_epochs = 0
#do train with batch_size full examples in one training
batch_size= 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epochs in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch) :
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict = {x:batch_x, y:batch_y})
            avg_cost += c/total_batch
            
        print('Epoch:', '%04d' % (epochs + 1), 'cost = ', '{:.9f}'.format(avg_cost))
        
        print("Accuracy: ", accuracy.eval(session=sess, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
    #test model with random picked image
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction:", sess.run(tf.argmax(h,1), feed_dict = {x:mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()