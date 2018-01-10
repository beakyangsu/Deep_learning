# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:15:46 2018

@author: yangsu
"""

import numpy as np
import tensorflow as tf

image = np.array([[[[4], [3]],
                   [[2], [1]]]], dtype=np.float32)

pool = tf.nn.max_pool(image, ksize = [1,2,2,1], strides=[1,1,1,1], padding='SAME')
print(pool.shape)
print(pool.eval())