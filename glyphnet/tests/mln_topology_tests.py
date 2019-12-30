#!/usr/bin/python3

import unittest

"""
Also see:
https://docs.python.org/3/library/unittest.html#basic-example
"""

class TestStringMethods(unittest.TestCase):
    def test1(self):
        print('found me 0')

if __name__ == '__main__':
    unittest.main()
    print('found me')
