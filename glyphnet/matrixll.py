#!/usr/bin/python3

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
            assert len(m[x]) == h, "Matrix not square(A), or incorrect expewcted h=%d (actual=%d at row %d)"%(h, len(m[x]), x)
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
            assert len(m[x]) == h, "Matrix not square(B), or incorrect expected h=%d (actual=%d at row %d)"%(h, len(m[x]), x)

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
