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
W = 2 #15
H = 2 #15
BATCHSIZE = 4 #2

# receptive field size
RF1 = 3

input = tf.placeholder(PIXEL_DTYPE, [None, W, H, RGB3DIMS])
#reshp = tf.reshape(input, [UNKNOWN_SIZE, W*H, RGB3DIMS])
#output = reshp * 2

ll = []
for x in range(W):
    for y in range(H):
            #for c in range(RGB3DIMS):
            ll += [input[:, x,y,:][:, None, :]]
layer_h1 = tf.concat(ll, axis=1) # row
output = layer_h1 * 2

print(input[0,0,0])
print(input[0,0,0,0])
print(input[-1,0,0,0]) # no
print(input[:, 0,0,0]) # Tensor("strided_slice_17:0", shape=(?,), dtype=uint8)

print(layer_h1)

#==================================================
# input data

# img_shape = (W,H, RGB3DIMS)

row123H = np.arange(H, dtype=np.uint8)
img_shape1H = (W, 1, RGB3DIMS)
data_img_1 = np.tile( row123H[None,:,None], img_shape1H)

row123W = np.arange(W, dtype=np.uint8)
img_shape1W = (1, H, RGB3DIMS)
data_img_2 = np.tile( row123W[:,None,None], img_shape1W)
data_img = data_img_1 + data_img_2*100

print(np.mean(data_img, axis=2))
data_images_batch = np.tile( data_img_1[None, :,:,:], [BATCHSIZE, 1,1,1])
print(np.mean(data_images_batch, axis=3))


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

