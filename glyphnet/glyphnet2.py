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

# PIXEL_DTYPE = tf.uint8
PIXEL_DTYPE = tf.float32
HL_DTYPE = tf.float32
WEIGHT_DTYPE = tf.float32


def make_conv_rf(input, SHAPE, RF1, nonlinearity1):
    (W,H,RGB3DIMS) = SHAPE
    assert tuple(input.shape[1:]) == (W,H,RGB3DIMS), """ explicitl specified SHAPE (size) must match %s. """ % (repr(input.shape[1:]))

    NEWSHAPE = (W-RF1+1,H-RF1+1,RGB3DIMS)
    assert W-RF1+1 > 0, """RF size %d does not fit in W=%d""" % (RF1, W)
    assert H-RF1+1 > 0, """RF size %d does not fit in H=%d""" % (RF1, H)

    assert RF1 > 1, """ no point in convolution with RF=%d < 2 """ %(RF1)

    ll = []
    for x in range(W-RF1+1):
        for y in range(H-RF1+1):
            #for c in range(RGB3DIMS):
            print('x,y', x,y)
            suminp = 0
            for dx in range(RF1):
                for dy in range(RF1):
                    print('x,y,dx,dy  ', x,y,dx,dy, '  + -> ', x+dx, y+dy)
                    inp_x, inp_y = x+dx, y+dy
                    v1 = input[:, inp_x,inp_y, :]
                    randinitval = tf.random_uniform([1], -1, 1, seed=0)
                    w1 = tf.Variable(randinitval, dtype=WEIGHT_DTYPE)
                    suminp = suminp + w1 * v1
            print('>>', suminp)
            b1 = tf.Variable(0.0, dtype=HL_DTYPE)  # (tf.zeros([1]) )
            suminp = suminp + b1
            out1 = nonlinearity1( suminp )

            #ll += [v1[:, None, :]]
            ll += [out1[:, None, :]] # prepare for row-like structure
    layer_h1 = tf.concat(ll, axis=1) # row: (W*H) x RGB3

    NEWRESHAPE = [-1, W-RF1+1,H-RF1+1,RGB3DIMS]
    reshaped_hidden_layer = tf.reshape(layer_h1, NEWRESHAPE)
    return reshaped_hidden_layer

# =================================================

# Fixme: the RGB needs to annihilate at level 1
# The level2 needs to have a quasi-location: based on which we define a distance.
#  Pne way is to have another factor instead of 3 of RGB. This means expantion of dimensions (as in LGN to V1)
RGB3DIMS = 1
W = 5 #15
H = 5 #15
BATCHSIZE = 4 #2

# receptive field size
RF1 = 4 #3
RF2 = 2

input = tf.placeholder(PIXEL_DTYPE, [None, W, H, RGB3DIMS])
#reshp = tf.reshape(input, [UNKNOWN_SIZE, W*H, RGB3DIMS])
#output = reshp * 2

nonlinearity1 = tf.nn.relu
#nonlinearity1 = tf.sigmoid
layer_h1 = make_conv_rf(input, (W,H,RGB3DIMS), RF1, tf.nn.relu)
layer_h2 = make_conv_rf(layer_h1, (W-RF1+1,H-RF1+1,RGB3DIMS), RF2, tf.nn.relu)

print(input.shape, '->', layer_h1.shape)
print(layer_h1.shape, '->', layer_h2.shape)

output = layer_h2 * 2

print(input[0,0,0])
print(input[0,0,0,0])
print(input[-1,0,0,0]) # no
print(input[:, 0,0,0]) # Tensor("strided_slice_17:0", shape=(?,), dtype=uint8)

print('layer_h1', layer_h1) # shape=(?, 6, 3) = (?, W*H, 3)

#==================================================
# input data
# img_shape = (W,H, RGB3DIMS)

def data_maker_fake(BATCHSIZE, shape, pixel_npdtype):
    (W,H,RGB3DIMS) = shape

    # numpy only
    row123H = np.arange(H, dtype=pixel_npdtype)
    img_shape1H = (W, 1, RGB3DIMS)
    data_img_1 = np.tile( row123H[None,:,None], img_shape1H)

    row123W = np.arange(W, dtype=pixel_npdtype)
    img_shape1W = (1, H, RGB3DIMS)
    data_img_2 = np.tile( row123W[:,None,None], img_shape1W)
    data_img = data_img_1 + data_img_2*100

    print(np.mean(data_img, axis=2))
    data_images_batch = np.tile( data_img_1[None, :,:,:], [BATCHSIZE, 1,1,1])
    print(np.mean(data_images_batch, axis=3))
    return data_images_batch

data_images_batch = data_maker_fake(BATCHSIZE, (W,H,RGB3DIMS), np.float)

#=================================================
# running the NN

sess = tf.Session()

sess.run( tf.global_variables_initializer() )

(out_data,) = \
    sess.run([output], feed_dict={input: data_images_batch})

# report the results
print('Input: ---------------')
print(data_images_batch)
print('Output: ---------------')
print(out_data)

