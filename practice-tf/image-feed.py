
# -*- coding: utf-8 -*-
#!/usr/bin/env python3


import tensorflow as tf
import numpy as np
import time
import scipy.misc
import imageio
np.set_printoptions(threshold=5, linewidth=80, precision=1)


RGB3DIMS = 1
W = 2
H = 3
BATCHSIZE = 2

# wiring up the circuit
input = tf.placeholder(tf.uint8, [BATCHSIZE, W, H, RGB3DIMS])
# input = tf.placeholder(tf.uint8, [None, W, H, RGB3DIMS])
output = input * 2

# input data

#data_img = np.zeros(img_shape, dtype=np.uint8)

img_shape = (W,H, RGB3DIMS)
img_shape1 = (W, 1, RGB3DIMS)
row = np.arange(H, dtype=np.uint8)
data_img = np.tile( row[None,:,None], img_shape1)

data_images_batch = np.tile( data_img[None, :,:,:], [BATCHSIZE, 1,1,1])



# running the NN

sess = tf.Session()
(out_data,) = \
    sess.run([output], feed_dict={input: data_images_batch})

# report the results
print('Input: ---------------')
print(data_images_batch)
print('Output: ---------------')
print(out_data)
