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
import math

from utils.pcolor import PColor

from utils import image_loader #import choose_random_batch
choose_random_batch = image_loader.choose_random_batch
from geo_maker import geometry_maker #import simple_triangles
simple_triangles = geometry_maker.simple_triangles

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

"""
    Maintains a 3D array as a list of list,
    with guarantee that all rows are of the same length
    (i.e. `h`)
"""
class matrixll:
    @staticmethod
    def iter_rows(m, h):
        assert isinstance(m, list)
        w = len(m)
        for x in range(w):
            row = m[x]
            assert isinstance(row, list)
            assert len(row) == h
            yield x, row

    @staticmethod
    def iter_elems(m, h):
        matrixll.check(m, -1, h)
        w = len(m)
        for x in range(w):
            assert isinstance(m[x], list)
            assert len(m[x]) == h
            for y in range(h):
                elem = m[x][y]
                yield x,y, elem

    @staticmethod
    def check(m, w1, h):
        assert isinstance(m, list)
        assert w1 == -1
        w = len(m)
        for x in range(w):
            assert isinstance(m[x], list)
            assert len(m[x]) == h
            for y in range(h):
                elem = m[x][y]
                assert (elem is None) or (elem is 1)

    @staticmethod
    def create_matrixll(w,h, default_value):
        # np.ndarray((w,h), dtype=int)
        m = []
        for x in range(w):
            m += [[]]
            for y in range(h):
                assert len(m[x]) == y
                m[x] += [default_value]
        return m

    @staticmethod
    def shape(m, h):
        w = len(m)
        if h == 'derive':
            if w == 0:
                raise Exception('cannot derive shape from a matrix of 0 rows: 0x?')
            h = len(m[0])
        matrixll.check(m,-1, h)
        return (w,h)

class MLNTopology():
    INDEX_INT_TYPE = int
    def __init__(self):
        # alt: tuple(int), xor, np.array(int), xor: simply an int !
        self.layers_shape = []  #  : List[int]

        # nominal coord system dimensions: e.g. (x,y,ch) (theta,sigma,x,y,hue) (feature_id)
        # i.e. for layer's intrinsic (functional) topology
        #  List[int]
        self.layers_coord_dims = []

        # connectivity_matrix: a sparse matrix of certain sort of indices. (address?)
        #   address: 1.tuple (of size len(shape))  2. string  3. raw_flat_index (int)
        self.matrices = []
        # some layers can (suggested) to be arranged in shapes/tensors

        # both map(v.) and a map (n.)
        self.coords_map = []

        self.consistency_invariance_check()

    def consistency_invariance_check(self):
        nl = len(self.layers_shape)
        nl2 = len(self.layers_coord_dims)
        nl3 = len(self.matrices)
        print('ml3',nl3, ' nl', nl)
        assert nl == nl2
        if nl == 0:
            assert nl3 == 0
        else:
            assert nl3 == nl-1
        for li in range(nl):
            neurons_count = self.layers_shape[li]
            assert isinstance(self.layers_shape[li], int)
            assert isinstance(self.layers_coord_dims[li], int)
            assert len(self.coords_map[li]) == neurons_count
            # assert len(self.layers_coord_dims[li]) > 0
            for ni in range(neurons_count):
                address = ni # address is simply the node (neuron) index, i.e. an `int`
                coords = self.coords_map[li][address]
                assert isinstance(coords, tuple)
                assert len(coords) == self.layers_coord_dims[li]

        assert len(self.matrices) == nl
        for cli in range(1, nl):
            this_layer = cli
            prev_layer = cli-1

            curr_shape = self.layers_shape[this_layer]
            prev_shape = self.layers_shape[prev_layer]
            assert isinstance(curr_shape, int)
            assert isinstance(prev_shape, int)
            #assert curre_shape[0] == prev_shape[1]
            #FIXME:
            assert self.matrices.shape == ()
            w = curr_shape
            h = prev_shape
            # self.matrices[layer] : List[List[int]]
            assert w == len(self.matrices[cli])
            matrixll.check(self.matrices[cli], w, h)

    def report(self, internals):
        nl = len(self.layers_shape)
        print('Report for topology (of %d layers):' % nl, self)
        print('   shape', self.layers_shape)
        print('   coords', self.layers_coord_dims)
        if internals:
            for li in range(nl):
                print('      layer: ', li, 'connections:', matrixll.shape(self.matrices[li], 'derive'))
                print('                        coords for %d entries' % len(self.coords_map[li]))
    def layer_num_elem(self, layer_no):
        numel = self.layers_shape[layer_no]
        assert isinstance(numel, int)
        return numel

    def add_layer(self, new_layer_shape, coord_dims):
        numnodes = new_layer_shape
        assert isinstance(numnodes, int)
        self.layers_shape += [new_layer_shape]
        self.layers_coord_dims += [coord_dims]
        self.coords_map += [[(i,) for i in range(numnodes)]]
        nl = len(self.layers_shape)
        if nl > 0:
            prev_layer_shape = self.layers_shape[-1]
            (w,h) = (np.prod(prev_layer_shape), np.prod(new_layer_shape))
            connectivity_matrix = matrixll.create_matrixll(w,h, None)
            print('connectivity_matrix', matrixll.shape(connectivity_matrix, h))
            self.matrices += [connectivity_matrix]
        self.report(True)
        self.consistency_invariance_check()

    def iterate_connections(self, prev_layer, this_layer):
        self.consistency_invariance_check()
        assert prev_layer == this_layer - 1
        (prev_layer, this_layer) = (this_layer - 1, this_layer)
        curr_shape = self.layers_shape[this_layer]
        prev_shape = self.layers_shape[prev_layer]
        w = curr_shape
        h = prev_shape
        assert isinstance(w, int)
        assert isinstance(h, int)

        for x in range(w):
            for y in range(h):
                matrix = self.matrices[prev_layer]
                # connection_object_ref
                conn_obj = matrix[x][y]
                if conn_obj is None:
                    continue
                (address1, address2) = (x,y)
                yield address1, address2, conn_obj


    def iterate_layers(self):
        self.consistency_invariance_check()
        nl = len(self.layers_shape)
        for i in range(nl):
            numel = self.layers_shape[nl]
            yield i, numel
            # yield i, numel, i+1, next_layer_shape

    def connect(self, prev_layer_no, address1_prev, address2_next, conn_obj):
        layer_no_next = prev_layer_no+1
        assert isinstance(address1_prev, int)
        assert isinstance(address2_next, int)
        # test
        self.get_node_metadata(layer_no_next, address2_next)
        self.get_node_metadata(prev_layer_no, address1_prev)
        matrix = self.matrices[prev_layer_no]
        assert matrix[address1_prev, address2_next] is None
        matrix[address1_prev][address2_next] = conn_obj
        assert conn_obj == 1
        self.consistency_invariance_check()

    # deprecated. all layers are flat
    """ shape dims """
    """
    def get_address_indices(layer_no):
        if layer_no == 1:
            return 2+1
        else:
            return 1
    """

    def get_node_metadata(self, layer, address):
        layer_no = layer
        assert layer_no >= 0 and layer_no < len(self.layers_shape)
        dims = self.layers_coord_dims[layer_no]
        #self.layer_dim_names[0] = ['x', 'y', 'ch']
        #return {x: , y:}
        assert isinstance(address, int)
        coords = self.coords_map[layer_no][address]
        assert len(coords) == dims
        return coords

    def get_layer_coord_system(self, layer_no):
        # typically: [3,1,1,1,...]  or [3,3,1,1,1,..]
        dims = self.layers_coord_dims[layer_no]
        return dims

def test_MLNTopology():
    expected_shapes = [15*15*3,1]
    expected_coords = [3,1]

    topology = MLNTopology()
    (W,H,ChRGB) = (15,15,3)
    topology.add_layer(W*H*ChRGB, 3)
    topology.consistency_invariance_check()
    topology.add_layer(128, 1)
    topology.consistency_invariance_check()
    for l, numel in topology.iterate_layers():
        dims = topology.get_layer_coord_system()
        assert expected_shapes[l] == numel
        assert expected_coords[l] == dims

test_MLNTopology()
print('fine')

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
