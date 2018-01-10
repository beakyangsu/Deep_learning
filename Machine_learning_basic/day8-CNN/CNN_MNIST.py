# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:19:27 2018

@author: yangsu
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
#read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

#(-1,28,28,1) = (None,28,28,1) and one color
img = mnist.train.images[0].reshape(28,28)
plt.imshow(img, cmap='gray')

img = img.reshape(-1,28,28,1)
#5-kernel 3x3, strides 2 : 28x28 ->14x14
w1 = tf.Variable(tf.random_normal([3,3,1,5], stddev=0.01))
conv2d = tf.nn.conv2d(img,w1,strides=[1,2,2,1], padding='SAME')
print(conv2d)

sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)

#output : 5 image
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')

#pooling max value in 2x2 section, stride 2 : 14x14 -> 7x7 
pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(pool)


pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)

#output : 5 image
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7,7), cmap='gray')

