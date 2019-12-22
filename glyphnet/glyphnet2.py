# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Version 2:
#   Relatively Sparse synapses: local connections only at Receptive Fields.


import tensorflow as tf
import numpy as np
import time
import scipy.misc
import imageio

from utils.pcolor import PColor

from utils import image_loader #import choose_random_batch
choose_random_batch = image_loader.choose_random_batch
from geo_maker import geometry_maker #import simple_triangles
simple_triangles = geometry_maker.simple_triangles

UNKNOWN_SIZE = -1



#=================================================

PIXEL_DTYPE = tf.uint8

RGB3DIMS = 3
W = 15
H = 15
BATCHSIZE = 2

# receptive field size
RF1 = 3

input = tf.placeholder(PIXEL_DTYPE, [None, W, H, RGB3DIMS])
#reshp = tf.reshape(input, [UNKNOWN_SIZE, W*H, RGB3DIMS])
#output = reshp * 2

j = 0
k = 0
ll = [input[:, i,j,k][:, None, None] for i in range(W)]
layer_h1 = tf.concat(ll, axis=1) # axis=0??
output = layer_h1 * 2

print(input[0,0,0])
print(input[0,0,0,0])
print(input[-1,0,0,0]) # no
print(input[:, 0,0,0]) # Tensor("strided_slice_17:0", shape=(?,), dtype=uint8)

print(layer_h1)

#==================================================
# input data

img_shape = (W,H, RGB3DIMS)
img_shape1 = (W, 1, RGB3DIMS)
row = np.arange(H, dtype=np.uint8)
data_img = np.tile( row[None,:,None], img_shape1)
data_images_batch = np.tile( data_img[None, :,:,:], [BATCHSIZE, 1,1,1])

#=================================================
# running the NN

sess = tf.Session()
(out_data,) = \
    sess.run([output], feed_dict={input: data_images_batch})

# report the results
print('Input: ---------------')
print(data_images_batch)
print('Output: ---------------')
print(out_data)

