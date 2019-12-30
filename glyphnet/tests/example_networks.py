#!/usr/bin/python3

import mln_topology
MLNTopology=mln_topology.MLNTopology
tuple_iter=mln_topology.tuple_iter

QUIET_TESTS = False

#from tests.example_networks import small_test_mlp

def small_test_mlp():
    t = MLNTopology()
    t.add_layer(1, 1, tuple_iter((1,)))
    t.add_layer(2, 1, tuple_iter((2,)))
    t.add_layer(3, 1, tuple_iter((3,)))
    if not QUIET_TESTS:
        t.report(True)

    conobj = 1
    t.connect(0, 0,1, conobj, check=True)
    t.connect(1, 1,2, conobj, check=True)
    t.connect(1, 1,1, conobj, check=True)
    t.consistency_invariance_check()
    return t

