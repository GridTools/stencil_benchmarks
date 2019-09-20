import unittest

import pandas as pd

from stencil_benchmarks.tools import common


class TestExpandTuples(unittest.TestCase):
    def test_without_tuples(self):
        df = pd.DataFrame([{'a': 0, 'b': 1}, {'a': 2, 'b': 3}])
        self.assertTrue(common.expand_tuples(df).equals(df))

    def test_with_tuples(self):
        df = pd.DataFrame([{
            'a': 0,
            'b': 1,
            'c': (2, 3),
            'd': (4, 5)
        }, {
            'a': 6,
            'b': 7,
            'c': (8, 9),
            'd': (0, 1)
        }])
        expected = pd.DataFrame([{
            'a': 0,
            'b': 1,
            'c-0': 2,
            'c-1': 3,
            'd-0': 4,
            'd-1': 5
        }, {
            'a': 6,
            'b': 7,
            'c-0': 8,
            'c-1': 9,
            'd-0': 0,
            'd-1': 1
        }])
        self.assertTrue(common.expand_tuples(df).equals(expected))
