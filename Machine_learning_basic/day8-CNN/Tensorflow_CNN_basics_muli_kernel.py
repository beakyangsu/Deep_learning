# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:08:56 2018

@author: yangsu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 22:49:53 2018

@author: yangsu
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
image = np.array([[[[1],[2],[3]],
                 [[4],[5],[6]],
                 [[7],[8],[9]]]], dtype = np.float32)

print("image.shape: ", image.shape)
plt.imshow(image.reshape(3,3), cmap='Greys')

# 3 kernel
weight = tf.constant([[[[1., 10., -1.]], [[1., 10., -1.]]],
                      [[[1., 10., -1.]], [[1., 10., -1.]]]])

print("weight.shape:", weight.shape)
conv2d = tf.nn.conv2d(image,weight, strides=[1,1,1,1], padding='SAME')
conv2d_img = conv2d.eval()
print("conv2d_image.shape :", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)

#output : 3 image
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')