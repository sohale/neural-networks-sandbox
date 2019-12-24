# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Version 2:
   Relatively Sparse synapses: local connections only at Receptive Fields.

To run:
    python glyphnet2.py; tensorboard --logdir="./graph"
"""

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


def make_conv_rf(input, SHAPE, RF1, nonlinearity1, lname):
    (W,H,RGB3DIMS) = SHAPE
    assert tuple(input.shape[1:]) == (W,H,RGB3DIMS), """ explicitl specified SHAPE (size) must match %s. """ % (repr(input.shape[1:]))

    #NEWSHAPE = (W-RF1+1,H-RF1+1,RGB3DIMS)
    #assert W-RF1+1 > 0, """RF size %d does not fit in W=%d""" % (RF1, W)
    #assert H-RF1+1 > 0, """RF size %d does not fit in H=%d""" % (RF1, H)

    assert RF1 > 1, """ no point in convolution with RF=%d < 2 """ %(RF1)

    # classes of variables:
    #   * trainable:          at AdamOptimiser()
    #   * placeholder:        at sess.run()
    #   * initialisation-time assignable (at sess.run() via tf.global_variables_initializer()  )

    with tf.variable_scope('L'+lname):
      ll = []
      # (x,y) are the unique indices of the output
      # note: the smaller range has nothing to do with the next layer shrinking smaller.
      # the output unit needs to have a center-of-RF, based on which we reduce the layer. We depart from convensional, just here.
      # (x,y) is also the coords of the location.
      for x in range(W):
        for y in range(H):
          cuname1 = "x%dy%d"%(x,y)
          with tf.variable_scope(cuname1):
            #for c in range(RGB3DIMS):
            suminp = None
            for dx in range(RF1):
                for dy in range(RF1):
                    # print('x,y,dx,dy  ', x,y,dx,dy, '  + -> ', x+dx, y+dy)
                    # coords and index on input layer.
                    inp_x, inp_y = x+dx, y+dy
                    if inp_x < 0 or inp_x >= W:
                        continue
                    assert W == input.shape[1]
                    if inp_y < 0 or inp_y >= H:
                        continue
                    assert H == input.shape[2]
                    v1 = input[:, inp_x,inp_y, :]
                    randinitval = tf.random_uniform([1], -1, 1, seed=0)  # doesn accept trainable=False,
                    w1 = tf.Variable(initial_value=randinitval, trainable=True, dtype=WEIGHT_DTYPE)
                    if suminp is None:
                        suminp = w1 * v1
                    else:
                        suminp = suminp + w1 * v1
            b1 = tf.Variable(initial_value=0.0, trainable=True, dtype=HL_DTYPE)
            suminp = suminp + b1
            conv_unit_outp = nonlinearity1( suminp )
            # Why in tensorboard, the outputs are not of similar type? also arrows output `input` seem incorrect.

            #ll += [v1[:, None, :]]
            ll += [conv_unit_outp[:, None, :]] # prepare for row-like structure
      layer_h1 = tf.concat(ll, axis=1) # row: (W*H) x RGB3

      NEWRESHAPE = [-1, W-RF1+1,H-RF1+1,RGB3DIMS]
      reshaped_hidden_layer = tf.reshape(layer_h1, NEWRESHAPE)
    return reshaped_hidden_layer

    # why input has 54 outputs, while there are 25 elements only.
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

input = tf.placeholder(PIXEL_DTYPE, [None, W, H, RGB3DIMS], name='i1')
#reshp = tf.reshape(input, [UNKNOWN_SIZE, W*H, RGB3DIMS])
#output = reshp * 2

nonlinearity1 = tf.nn.relu
#nonlinearity1 = tf.sigmoid
layer_h1 = make_conv_rf(input, (W,H,RGB3DIMS), RF1, tf.nn.relu, lname='H1')
layer_h2 = make_conv_rf(layer_h1, (W-RF1+1,H-RF1+1,RGB3DIMS), RF2, tf.nn.relu, lname='H2')



if False:
    print(input.shape, '->', layer_h1.shape)
    print(layer_h1.shape, '->', layer_h2.shape)

    print(input[0,0,0])
    print(input[0,0,0,0])
    print(input[-1,0,0,0]) # no
    print(input[:, 0,0,0]) # Tensor("strided_slice_17:0", shape=(?,), dtype=uint8)

output = layer_h2 * 2
if False:
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

def data_maker_geometrik(BATCHSIZE, shape, pixel_npdtype):
    W_H = shape[:2]
    (W,H) = W_H
    WxH = W*H  # np.prod(np.array([W,H]))
    R3 = shape[2]
    images_batch__list = simple_triangles(WxH, R3, W_H, how_many_samples=BATCHSIZE)
    images_training_batch = np.stack(images_batch__list, axis=0)
    #assert (FLATTENED_SIZE,) == images_training_batch.shape[1:]
    return np.reshape(images_training_batch, [BATCHSIZE, W,H,R3])

# data_images_batch = data_maker_fake(BATCHSIZE, (W,H,RGB3DIMS), np.float)
data_images_batch = data_maker_geometrik(BATCHSIZE, (W,H,RGB3DIMS), np.float)

print('(W,H,RGB3DIMS)', (W,H,RGB3DIMS))

#=================================================
# running the NN

sess = tf.Session()

# For Tesnorboard. tensorboard --logdir="./graph" # http://localhost:6006/
graph_writer = tf.summary.FileWriter("./graph/", sess.graph)


print('tf.global_variables_initializer():', tf.global_variables_initializer())
sess.run( tf.global_variables_initializer() )

(out_data,) = \
    sess.run([output], feed_dict={input: data_images_batch})

graph_writer.close()


# report the results
print('Input: ---------------')
print(data_images_batch)
print('Output: ---------------')
print(out_data)

