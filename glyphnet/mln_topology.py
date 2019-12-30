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
        # rename: coord_dims[]
        self.layers_coord_dims = []


        # connectivity_matrix: a sparse matrix of certain sort of indices. (address?)
        #   address: 1.tuple (of size len(shape))  2. string  3. raw_flat_index (int)
        # rename: conn_matrixll[]
        self.matrices = []
        # some layers can (suggested) to be arranged in shapes/tensors

        # "map" as both map(v.) and a map (n.)
        # rename: nodes_coords
        self.coords_map = []

        self.consistency_invariance_check()

    def consistency_invariance_check(self):
        nl = len(self.layers_shape)
        nl2 = len(self.layers_coord_dims)
        nl3 = len(self.matrices)
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

            # todo: if thorough: check len(coords) == self.layers_coord_dims[li]

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
            assert w == len(m)
            matrixll.check(m, -1, h)

    def create_reverse(self):
        rev = MLNTopology()
        nl = len(self.layers_shape)
        for li1 in reversed(range(nl)):
            lir = nl-1 - li1
            new_layer_shape = self.layers_shape[li1]
            coord_dims = self.layers_coord_dims[li1]
            coord_iterator = self.coords_map[li1]
            rev.add_layer(new_layer_shape, coord_dims, coord_iterator)

        for li in reversed(range(1,nl)):
            lfrom1 = li-1
            lto1 = li
            lir = nl-1 - li
            for (ifrom, ito, conn_obj) in self.iterate_connections(lfrom1, lto1):
                rev.connect(lir, ito, ifrom, conn_obj, check=False)
        rev.consistency_invariance_check()
        return rev

    @staticmethod
    def encode_matrixll(mat):
        return repr(mat)

    @staticmethod
    def size_string(int_tuple):
        return 'x'.join([str(i) for i in int_tuple])

    def all_details(self):
        payload = []
        payload.append(type(self).__name__)

        payload.append(repr(self.layers_shape)) # shape
        payload.append(repr(self.layers_coord_dims)) # coord_dims

        payload.append('connections:')
        nl = len(self.layers_shape)
        payload.append('%d layers'% nl)
        for li in range(nl-1):
            m = self.matrices[li]
            payload.append(MLNTopology.size_string(matrixll.shape(m, 'derive')))
            payload.append(MLNTopology.encode_matrixll(m))

        payload.append('coords:')

        for li in range(nl):
            coords = self.coords_map[li]
            payload.append( repr(coords) )

        for i in range(len(payload)):
            assert isinstance(payload[i], str), str(i) + ':' + repr(payload[i])

        return '\n'.join(payload)

    def report(self, internals):
        nl = len(self.layers_shape)
        indent = '   '
        indent2 = '      '
        print('Report for topology (of %d layers):' % nl, self)
        print(indent, 'shape', self.layers_shape)
        print(indent, 'coords', self.layers_coord_dims)
        if internals:
            #print('self.matrices', len(self.matrices), self.matrices) # too long
            #print('self.coords_map', len(self.coords_map), self.coords_map) # too long
            for li in range(nl-1): # iterate connection matrices
                m = self.matrices[li]
                print(indent2, 'm:', len(m))
                print(indent2, 'm[0]:', len(m[0]))
                print(indent2,'layer: ', li, 'connections:', matrixll.shape(m, 'derive'))
            for li in range(nl): # iterate layers
                print(indent2,'self.coords_map[li]', len(self.coords_map[li]))
                print(indent2,'coords for %d entries' % len(self.coords_map[li]))

    # rename: nodes_count()
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
        assert len(self.coords_map[-1]) == self.layers_shape[-1], 'inconsistent number of coord tuples provided'
        if nl > 0:
            prev_layer_shape = self.layers_shape[-2]
            (w,h) = (np.prod(prev_layer_shape), np.prod(new_layer_shape))
            connectivity_matrix = matrixll.create_matrixll(w,h, None)
            assert matrixll.shape(connectivity_matrix, 'derive') == (w,h)
            self.matrices += [connectivity_matrix]

        if not QUIET_TESTS:
            self.report(True)
        self.consistency_invariance_check()

    def iterate_connections(self, prev_layer, this_layer):
        self.consistency_invariance_check()
        assert prev_layer == this_layer - 1
        assert prev_layer >= 0
        assert this_layer < len(self.layers_shape)
        (prev_layer, this_layer) = (this_layer - 1, this_layer)
        next_shape = self.layers_shape[this_layer]
        prev_shape = self.layers_shape[prev_layer]
        w = prev_shape
        h = next_shape
        assert isinstance(w, int)
        assert isinstance(h, int)

        matrix = self.matrices[prev_layer]
        d = matrixll.shape(matrix, 'derive')
        assert d == (w,h)

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

    def connect(self, prev_layer_no, address1_prev, address2_next, conn_obj, check=True):
        layer_no_next = prev_layer_no+1
        assert isinstance(address1_prev, int)
        assert isinstance(address2_next, int)
        # test
        self.get_node_metadata(layer_no_next, address2_next)
        self.get_node_metadata(prev_layer_no, address1_prev)
        matrix = self.matrices[prev_layer_no]
        assert matrix[address1_prev][address2_next] is None
        matrix[address1_prev][address2_next] = conn_obj
        assert conn_obj == 1
        if check:
            self.consistency_invariance_check()

    """
    Uses synaptic prune rule for connectivity:
    prune_rule = synaptic_prune_rule
    Arrows from lower layer index towards higher
    """
    def connect_all(self, prev_layer_no, next_layer_no, prune_rule):
        assert next_layer_no == prev_layer_no+1, 'only MLP-style is allowed: connections must between consecutive layers only'
        next_shape_int = self.layers_shape[next_layer_no]
        prev_shape_int = self.layers_shape[prev_layer_no]
        coords_next = self.coords_map[next_layer_no]
        coords_prev = self.coords_map[prev_layer_no]
        assert isinstance(next_shape_int, int)
        for i_next in range(next_shape_int):
            # prev_layer_count = prev_shape_int
            for j_prev in range(prev_shape_int):
                coord_next = coords_next[i_next]
                coord_prev = coords_prev[j_prev]
                # apply synaptic prune rule for connectivity:
                if not prune_rule(coord_prev, coord_next):
                    conn_obj = 1
                    self.connect(prev_layer_no, j_prev, i_next, conn_obj, check=False)
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
        assert layer_no >= 0, "non-existing layer %d" % layer_no
        assert layer_no < len(self.layers_shape), "non-existing layer %d" % layer_no
        dims = self.layers_coord_dims[layer_no]
        #self.layer_dim_names[0] = ['x', 'y', 'ch']
        #return {x: , y:}
        assert isinstance(address, int)
        coords = self.coords_map[layer_no][address]
        assert len(coords) == dims
        return coords

    """
    # iterate over nodes
    def iter_node(self, layer_no):
        assert layer_no >= 0
        neurons_count = self.layers_shape[layer_no]
        for ni in range(neurons_count):
            yield \
                ni, \
                self.matrices[layer_no], \
                self.coords_map[layer_no]
    """

    # simple maping htat does not involve inf about conections
    def coord_map(self, layer_no, coords_map_lambda, newdims):
        assert layer_no >= 0
        assert isinstance(newdims, int)
        assert isinstance(coords_map_lambda, type(lambda:0))
        neurons_count = self.layers_shape[layer_no]
        # iterate over nodes
        for ni in range(neurons_count):
            coords = self.coords_map[layer_no][ni]
            #assert len(coords) == self.layers_coord_dims[li]
            assert isinstance(coords, tuple)
            coords_new = coords_map_lambda(coords)
            assert isinstance(coords_new, tuple)
            assert len(coords_new) == newdims
            self.coords_map[layer_no][ni] = coords_new
        self.layers_coord_dims[layer_no] = newdims

    def get_layer_coord_system(self, layer_no):
        # typically: [3,1,1,1,...]  or [3,3,1,1,1,..]
        dims = self.layers_coord_dims[layer_no]
        return dims

def print0(*args):
    print('print0', args)
    return 0

# utilities
def connect_based_on_distance(topo, prev_layer_no, next_layer_no, radius):
    (_X, _Y, _RGB) = (0,1,2)
    # radius = 3.0
    topo.connect_all(prev_layer_no, next_layer_no,
        lambda coord1, coord2:
            (coord1[_X] - coord2[_X]) ** 2 +
            (coord1[_Y] - coord2[_Y]) ** 2
            >
            radius ** 2
    )


