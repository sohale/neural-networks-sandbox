#!/usr/bin/python3

import numpy as np
from matrixll import matrixll

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

        # "map" as both map(v.) and a map (n.)
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
            #print('a', len(self.coords_map[li]), neurons_count)
            assert len(self.coords_map[li]) == neurons_count
            # assert len(self.layers_coord_dims[li]) > 0
            for ni in range(neurons_count):
                address = ni # address is simply the node (neuron) index, i.e. an `int`
                coords = self.coords_map[li][address]
                assert isinstance(coords, tuple)
                #print(len(coords), self.layers_coord_dims[li])
                assert len(coords) == self.layers_coord_dims[li]

        if nl > 0:
            assert len(self.matrices) == nl-1
        for cli in range(1, nl):
            next_layer = cli
            prev_layer = cli-1

            next_shape = self.layers_shape[next_layer]
            prev_shape = self.layers_shape[prev_layer]
            assert isinstance(next_shape, int)
            assert isinstance(prev_shape, int)
            h = next_shape
            w = prev_shape
            # self.matrices[layer] : List[List[int]]
            m = self.matrices[prev_layer]
            print(w, '??==', len(m), matrixll.shape(m, 'derive'))
            assert w == len(m)
            matrixll.check(m, -1, h)

    def report(self, internals):
        nl = len(self.layers_shape)
        print('Report for topology (of %d layers):' % nl, self)
        print('   shape', self.layers_shape)
        print('   coords', self.layers_coord_dims)
        if internals:
            #print('self.matrices', len(self.matrices), self.matrices) # too long
            #print('self.coords_map', len(self.coords_map), self.coords_map) # too long
            for li in range(nl-1): # iterate connection matrices
                m = self.matrices[li]
                print('m:', len(m))
                print('m[0]:', len(m[0]))
                print('      layer: ', li, 'connections:', matrixll.shape(m, 'derive'))
            for li in range(nl): # iterate layers
                print('        self.coords_map[li]', len(self.coords_map[li]))
                print('                        coords for %d entries' % len(self.coords_map[li]))
    def layer_num_elem(self, layer_no):
        numel = self.layers_shape[layer_no]
        assert isinstance(numel, int)
        return numel

    def add_layer(self, new_layer_shape, coord_dims, coord_iterator):
        numnodes = new_layer_shape
        assert isinstance(numnodes, int)
        nl = len(self.layers_shape)
        self.layers_shape += [new_layer_shape]
        self.layers_coord_dims += [coord_dims]
        self.coords_map += [None]
        self.coords_map[-1] = [tpl for tpl in coord_iterator]
         # [[(i,) for i in range(numnodes)]]
        if nl > 0:
            prev_layer_shape = self.layers_shape[-2]
            (w,h) = (np.prod(prev_layer_shape), np.prod(new_layer_shape))
            connectivity_matrix = matrixll.create_matrixll(w,h, None)
            print('connectivity_matrix', matrixll.shape(connectivity_matrix, h))
            assert matrixll.shape(connectivity_matrix, 'derive') == (w,h)
            self.matrices += [connectivity_matrix]
        self.report(True)
        self.consistency_invariance_check()

    def iterate_connections(self, prev_layer, this_layer):
        self.consistency_invariance_check()
        assert prev_layer == this_layer - 1
        (prev_layer, this_layer) = (this_layer - 1, this_layer)
        next_shape = self.layers_shape[this_layer]
        prev_shape = self.layers_shape[prev_layer]
        w = next_shape
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
            numel = self.layers_shape[i]
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
    expected_shapes = [15*15*3, 128, 15*15*3]
    expected_coords = [1, 1, 3]

    topology = MLNTopology()
    (W,H,ChRGB) = (15,15,3)
    topology.add_layer(W*H*ChRGB, 1, tuple_iter((W*H*ChRGB,)))
    topology.consistency_invariance_check()
    topology.add_layer(128, 1, tuple_iter((128,)))
    topology.consistency_invariance_check()

    for c in tuple_iter((W, H, ChRGB)):
        print(c)
    topology.add_layer(W*H*ChRGB, 3, tuple_iter((W, H, ChRGB)))
    topology.consistency_invariance_check()

    for l, numel in topology.iterate_layers():
        dims = topology.get_layer_coord_system(l)
        assert expected_shapes[l] == numel
        assert expected_coords[l] == dims


""" Iterated over indices of a tensor with given shape """
def tuple_iter(triple, prefix=()):
    #(W,H,ChRGB) = triple
    assert isinstance(triple, tuple)
    if len(triple) == 0:
        raise Exception('use tuple of len > 0')
    if len(triple) == 1:
        dim1 = triple[0]
        for i in range(dim1):
            yield tuple(prefix) + (i,)
        return
    dim1 = triple[0]
    for i in range(dim1):
        for y in tuple_iter((triple[1:]), prefix=prefix + (i,)):
            #yield (i,) + y
            yield y

def test_tuple_iter():
    def test_tuple_iter_case(shape, expected):
        actual = [tup for tup in tuple_iter(shape)]
        print('actual', actual)
        print('assert', repr(actual), '==', repr(expected))
        assert repr(actual) == repr(expected)
    test_tuple_iter_case((1,), [(0,)])
    test_tuple_iter_case((1,1), [(0,0)])
    test_tuple_iter_case((1,1,1), [(0,0,0)])
    test_tuple_iter_case((2,2,1), [(0,0,0), (0,1,0), (1,0,0), (1,1,0)])
    test_tuple_iter_case((1,1,3), [(0,0,0), (0,0,1), (0,0,2)])

test_tuple_iter()
test_MLNTopology()
print('unit tests passed. fine')
