#!/usr/bin/python3

import unittest

"""
Also see:
https://docs.python.org/3/library/unittest.html#basic-example
"""

import mln_topology
MLNTopology=mln_topology.MLNTopology

QUIET_TESTS = False

def __reshape_from_index(shape_tuple, val):
    #return __reshape_from_index(shape_tuple[])
    answer = []
    for ii in range(len(shape_tuple)-1,-1,-1):
        nd = shape_tuple[ii]
        r = val % nd
        val = int(val / nd)
        answer = [r] + answer
    assert val == 0
    return tuple(answer)

""" private method.
    param: i_tuple1 is a tuple of length 1 containning an int """
def __tuple_from_shape1(shape, i_tuple1):
    #return lambda idx: __reshape_from_index(shape, i_tuple1[0]) + lassert(len(i_tuple1) == 1)
    assert len(i_tuple1) == 1
    t = __reshape_from_index(shape, i_tuple1[0])
    return t

# makes coord maps DSL-ish
def lambda_from_shape(shape):
    return lambda i_tuple1: __tuple_from_shape1(shape, i_tuple1)

def small_test_mlp():
    t = MLNTopology()
    t.add_layer(1, 1, tuple_iter((1,)))
    t.add_layer(2, 1, tuple_iter((2,)))
    t.add_layer(3, 1, tuple_iter((3,)))
    conobj = 1
    t.connect(0, 0,1, conobj, check=True)
    t.connect(1, 1,2, conobj, check=True)
    t.connect(1, 1,1, conobj, check=True)
    t.consistency_invariance_check()
    return t

def test_MLNTopology():
    expected_shapes = [15*15*3, 64, 15*15*3]
    expected_coords = [1, 1, 2]

    topology = MLNTopology()
    (W,H,ChRGB) = (15,15,3)
    topology.add_layer(W*H*ChRGB, 1, tuple_iter((W*H*ChRGB,)))
    topology.consistency_invariance_check()
    topology.add_layer(64, 1, tuple_iter((64,)))
    topology.consistency_invariance_check()

    #for c in tuple_iter((W, H, ChRGB)):
    #    print(c)
    topology.add_layer(W*H*ChRGB, 3, tuple_iter((W, H, ChRGB)))
    topology.consistency_invariance_check()

    assert topology.coords_map[2][0] == (0,0,0)
    assert topology.coords_map[2][3] == (0,1,0)
    topology.coord_map(2,
        lambda xyc: ((xyc[0]+1)*100 + (xyc[1]+1)*10 + (xyc[2]+1)*1, xyc[2]+1),
        newdims=2)
    assert topology.coords_map[2][0] == (111, 1)
    assert topology.coords_map[2][3] == (121, 1)

    for l, numel in topology.iterate_layers():
        dims = topology.get_layer_coord_system(l)
        assert expected_shapes[l] == numel
        assert expected_coords[l] == dims

    for i,j,_ in topology.iterate_connections(0,1):
        raise Exception('no connection should exist yet')
    for i,j,_ in topology.iterate_connections(1,2):
        raise Exception('no connection should exist yet')

    topology.connect_all(1,2, lambda c1,c2: True )

    topology.coord_map(1,
        lambda i:
            (int(i[0]/8)*2.0, (i[0]%8)*2.0
                #+ print0(repr(i), type(i[0]), i[0], int(i[0]/8), i[0]%8)
            ),
        newdims=2
    )
    #def lassert(cond):
    #    assert cond, "lassert failed"
    #    return 0

    assert lambda_from_shape((2,2,2))((0,)) == (0,0,0)
    assert lambda_from_shape((8,8))((63,)) == (7,7)
    assert lambda_from_shape((15,15,3))((0,)) == (0,0,0)
    assert lambda_from_shape((15,15,3))((15*15*3-1,)) == (14,14,2)

    topology.coord_map(0,
        lambda_from_shape((15,15,3)),
        newdims=3
    )

    connect_based_on_distance(topology, 0,1, 3.0)

    if not QUIET_TESTS:
        ctr1 = 0
        for i,j,_ in topology.iterate_connections(0,1):
            ctr1 += 1

        ctr2 = 0
        for i,j,_ in topology.iterate_connections(1,2):
            ctr2 += 1
        print('ctr1', ctr1, 'of',
            topology.layer_num_elem(0) * topology.layer_num_elem(1) )
        print('ctr2', ctr2)

    # unit test for create_reverse()
    def test_create_reverse(t):
        encoded_expected = t.all_details()
        rev = t.create_reverse()
        revrev = rev.create_reverse()
        encoded2 = revrev.all_details()
        assert encoded2 == encoded_expected, "failed unit test for create_reverse()"
        nl = len(t.layers_shape)
        for i in range(nl):
            assert t.layers_shape[i] == rev.layers_shape[nl-i-1]
        assert encoded2 == encoded_expected, "failed unit test for create_reverse()"
        return [rev, revrev]

    test_create_reverse(topology)

    t = small_test_mlp()
    test_create_reverse(t)
    (rev, revrev) = test_create_reverse(t)
    if not QUIET_TESTS:
        print('------------original:')
        print(t.all_details())
        print('------------rev:')
        print(rev.all_details())
        print('------------rev-rev:')
        print(revrev.all_details())

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
        #print('actual', actual)
        #print('assert', repr(actual), '==', repr(expected))
        assert repr(actual) == repr(expected)

    test_tuple_iter_case((1,), [(0,)])
    test_tuple_iter_case((1,1), [(0,0)])
    test_tuple_iter_case((1,1,1), [(0,0,0)])
    test_tuple_iter_case((2,2,1), [(0,0,0), (0,1,0), (1,0,0), (1,1,0)])
    test_tuple_iter_case((1,1,3), [(0,0,0), (0,0,1), (0,0,2)])

class TestStringMethods(unittest.TestCase):
    def test1(self):
        print('found me0')

    def test_x(self):
        test_tuple_iter()

    def test_y(self):
        test_MLNTopology()

if __name__ == '__main__':
    # test_tuple_iter()
    # test_MLNTopology()
    # print('unit tests passed. fine')

    unittest.main()
    print('unit main')
