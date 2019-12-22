
# -*- coding: utf-8 -*-
#!/usr/bin/env python3


import tensorflow as tf
import numpy as np
import time
import scipy.misc
import imageio


RGB3DIMS = 3


input = tf.placeholder(tf.uint8, [None, None, RGB3DIMS])
output = input * 2

img_shape = (2,4,3)
img_shape1 = (img_shape[0], 1, img_shape[2])
#data_im1 = np.zeros(img_shape, dtype=np.uint8)
row = np.arange(img_shape[1], dtype=np.uint8) [None,:,None]
data_im1 = np.tile( row, img_shape1)

print(row.shape)
print(data_im1.shape)

np.set_printoptions(threshold=5, linewidth=80, precision=1)


sess = tf.Session()
(out_data,) = \
    sess.run([output], feed_dict={input: data_im1})

print('input')
print(data_im1)
print('output')
print(out_data)
