
# -*- coding: utf-8 -*-
#!/usr/bin/env python3


import tensorflow as tf
import numpy as np
import time
import scipy.misc
import imageio
np.set_printoptions(threshold=5, linewidth=80, precision=1)


RGB3DIMS = 3
W = 2
H = 4

# wiring up the circuit
input = tf.placeholder(tf.uint8, [None, None, RGB3DIMS])
output = input * 2

# input data

#data_im1 = np.zeros(img_shape, dtype=np.uint8)

img_shape = (W,H, RGB3DIMS)
img_shape1 = (W, 1, RGB3DIMS)
row = np.arange(H, dtype=np.uint8) [None,:,None]
data_im1 = np.tile( row, img_shape1)



# running the NN

sess = tf.Session()
(out_data,) = \
    sess.run([output], feed_dict={input: data_im1})

# report the results
print('input')
print(data_im1)
print('output')
print(out_data)
