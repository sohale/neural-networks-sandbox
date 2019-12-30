# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Version 2:
   Relatively Sparse synapses: local connections only at Receptive Fields.

To run:
    python glyphnet2.py; tensorboard --logdir="./graph"
"""
import mln_topology
MLNTopology = mln_topology.MLNTopology
exit()

import wireup_mltopology

exit()

import tensorflow as tf
import numpy as np
import time
import scipy.misc
import imageio
import math

from utils.pcolor import PColor

from utils import image_loader #import choose_random_batch
choose_random_batch = image_loader.choose_random_batch
from geo_maker import geometry_maker #import simple_triangles
simple_triangles = geometry_maker.simple_triangles

#from mln_topology import MLNTopology
import mln_topology
MLNTopology = mln_topology.MLNTopology

import wireup_mltopology


UNKNOWN_SIZE = -1


# ====================================
metadata_base = {}
def get_element_metadata(layer_obj, inp_x,inp_y):
    assert isinstance(inp_x, int)
    assert isinstance(inp_y, int)
    d = metadata_base[layer_obj]
    key = (inp_x,inp_y)
    metadata_content = d[key]
    if metadata_content is None:
        raise Exception("not found metadata enrty %r for %r" % (key, layer_obj))
    return metadata_content

def set_element_metadata(layer_obj, inp_x,inp_y, metadata_content):
    assert isinstance(inp_x, int)
    assert isinstance(inp_y, int)
    if layer_obj not in metadata_base: # first time
        metadata_base[layer_obj] = {}
    d = metadata_base[layer_obj]
    key = (inp_x,inp_y)
    d[key] = metadata_content

"""
Annotating each unit with coordinates'
"""
def set_metadata_bulk(W,H, input):
    assert isinstance(W, int)
    assert isinstance(H, int)
    for x in range(W):
        for y in range(H):
            mdata = (x,y, 1)
            set_element_metadata(input, x, y, mdata)

#=================================================

# PIXEL_DTYPE = tf.uint8
PIXEL_DTYPE = tf.float32
HL_DTYPE = tf.float32
WEIGHT_DTYPE = tf.float32

global weights_count
weights_count = 0


def make_conv_rf(input, INPUT_SHAPE, conv_spread_range, stride_xy, nonlinearity1, lname):
    # conv_spread_range = conv_offset_range
    (W,H,RGB3DIMS) = INPUT_SHAPE
    #print('input.shape for', lname, input.shape, ' asserting', tuple(input.shape[1:]), '==', (W,H,RGB3DIMS))
    assert tuple(input.shape[1:]) == (W,H,RGB3DIMS), """ explicitl specified INPUT_SHAPE (size) = %r must match %s. """ % (INPUT_SHAPE, repr(input.shape[1:]))
    global weights_count

    #print('OUT_RANGE', OUT_RANGE)
    #(Wout, Hout, chans_out) = OUT_RANGE
    Wout = int((W + stride_xy[0]-1)/stride_xy[0])*stride_xy[0]
    Hout = int((H + stride_xy[1]-1)/stride_xy[1])*stride_xy[1]
    # RF1 -> conv_spread_range

    assert isinstance(INPUT_SHAPE[0], int)
    assert isinstance(INPUT_SHAPE[1], int)
    assert isinstance(INPUT_SHAPE[2], int)
    assert isinstance(stride_xy[0], int)
    assert isinstance(stride_xy[1], int)
    assert len(stride_xy) == 2
    assert isinstance(conv_spread_range[0], int)
    assert isinstance(conv_spread_range[1], int)
    assert conv_spread_range[1]+1 - conv_spread_range[0] >= 1
    assert len(conv_spread_range) == 2

    assert tuple(input.shape[1:]) == INPUT_SHAPE
    assert int(input.shape[1]) == INPUT_SHAPE[0] # W
    assert int(input.shape[2]) == INPUT_SHAPE[1] # H
    assert int(input.shape[3]) == INPUT_SHAPE[2] # RGB3

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
      for x in range(0, Wout, stride_xy[0]):
        for y in range(0, Hout, stride_xy[1]):
          cuname1 = "x%dy%d"%(x,y)
          with tf.variable_scope(cuname1):
            #for c in range(RGB3DIMS):
            #suminp = None
            suminp = 0.0
            for dx in range(conv_spread_range[0], conv_spread_range[1]+1, 1):
                for dy in range(conv_spread_range[0], conv_spread_range[1]+1, 1):
                    # coords and index on input layer.
                    inp_x, inp_y = x+dx, y+dy
                    if inp_x < 0 or inp_x >= W:
                        continue
                    assert W == input.shape[1]
                    if inp_y < 0 or inp_y >= H:
                        continue
                    assert H == input.shape[2]
                    v1 = input[:, inp_x,inp_y, :]
                    (x,y,c) = get_element_metadata(input, inp_x,inp_y)
                    randinitval = tf.random_uniform([1], -1, 1, seed=0)  # doesn accept trainable=False,
                    w1 = tf.Variable(initial_value=randinitval, trainable=True, dtype=WEIGHT_DTYPE)
                    weights_count += 1
                    #if suminp is None:
                    #    suminp = w1 * v1
                    #else:
                    if True:
                        suminp = suminp + w1 * v1
            b1 = tf.Variable(initial_value=0.0, trainable=True, dtype=HL_DTYPE)
            suminp = suminp + b1
            conv_unit_outp = nonlinearity1( suminp )
            # Why in tensorboard, the outputs are not of similar type? also arrows output `input` seem incorrect.

            #ll += [v1[:, None, :]]
            ll += [conv_unit_outp[:, None, :]] # prepare for row-like structure
      layer_h1 = tf.concat(ll, axis=1) # row: (W*H) x RGB3

      NEWRESHAPE = [-1, int(Wout/stride_xy[0]), int(Hout/stride_xy[1]), RGB3DIMS]
      reshaped_hidden_layer = tf.reshape(layer_h1, NEWRESHAPE)

      set_metadata_bulk(W,H, reshaped_hidden_layer)

    return reshaped_hidden_layer

    # why input has 54 outputs, while there are 25 elements only.


# =================================================
from typing import List
from typing import TypeVar, Generic



def build_tf_network(topology: MLNTopology):
    pass
# =================================================

# Fixme: the RGB needs to annihilate at level 1
# The level2 needs to have a quasi-location: based on which we define a distance.
#  Pne way is to have another factor instead of 3 of RGB. This means expantion of dimensions (as in LGN to V1)
RGB3DIMS = 1
W = 5 #15
H = 5 #15
BATCHSIZE = 4 #2

# receptive field size
RF1 = int(4/2) #3
RF2 = int(2/2)

input = tf.placeholder(PIXEL_DTYPE, [None, W, H, RGB3DIMS], name='i1')
#reshp = tf.reshape(input, [UNKNOWN_SIZE, W*H, RGB3DIMS])
#output = reshp * 2
set_metadata_bulk(W,H, input)

weights_count = 0

nonlinearity1 = tf.nn.relu
#nonlinearity1 = tf.sigmoid
print('L1')
L0_SHAPE = (W,H,RGB3DIMS)
def shape_div(L0_SHAPE, m, n):
    W = int(L0_SHAPE[0])
    H = int(L0_SHAPE[1])
    CH = int(L0_SHAPE[2])
    return (math.ceil(W/m)*n, math.ceil(H/m)*n, CH)

#shape_div(L0_SHAPE,1,1),
layer_h1 = make_conv_rf(input, L0_SHAPE, (-RF1,RF1),   (1,1), tf.nn.relu, lname='H1')
print('L1>>>', layer_h1.shape)
L1_SHAPE = shape_div(layer_h1.shape[1:], 1,1)
#shape_div(L1_SHAPE,1,1),
layer_h2 = make_conv_rf(layer_h1, L1_SHAPE, (-3, 3),   (2,2), tf.nn.relu, lname='H2')
print('L2>>>', layer_h2.shape)
L2_SHAPE = shape_div(layer_h2.shape[1:], 1,1)
#L2OUT_SHAPE = shape_div(L2_SHAPE,2,1)  # shape_div(L0_SHAPE,2,2)
#shape_div(L2_SHAPE,2,1),
layer_h3 = make_conv_rf(layer_h2, L2_SHAPE, (-3, 3),   (2,2), tf.nn.relu, lname='H3')
print('L3>>>', layer_h3.shape)
L3_SHAPE = shape_div(layer_h3.shape[1:], 1,1) #shape_div(L0_SHAPE, 4,1)
#L3OUT_SHAPE = shape_div(L0_SHAPE,4,1) # shape_div(L0_SHAPE,4,4)
#shape_div(L3_SHAPE,2,1),
layer_h4 = make_conv_rf(layer_h3, L3_SHAPE, (-3, 3),   (2,2), tf.nn.relu, lname='H4')
print('L4>>>', layer_h4.shape)



if False:
    print(input.shape, '->', layer_h1.shape)
    print(layer_h1.shape, '->', layer_h2.shape)

    print(input[0,0,0])
    print(input[0,0,0,0])
    print(input[-1,0,0,0]) # no
    print(input[:, 0,0,0]) # Tensor("strided_slice_17:0", shape=(?,), dtype=uint8)

output = layer_h4 * 2
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


#print('tf.global_variables_initializer():', tf.global_variables_initializer())
sess.run( tf.global_variables_initializer() )

(out_data,) = \
    sess.run([output], feed_dict={input: data_images_batch})

graph_writer.close()


# report the results
print('Input: ---------------')
print(data_images_batch)
print('Output: ---------------')
print(out_data)

# 309
print('weights:', weights_count)
