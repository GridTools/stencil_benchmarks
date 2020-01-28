import unittest

import numpy as np

from stencil_benchmarks.tools import parallel


class TestRandomFill(unittest.TestCase):
    def test_1d(self):
        for dtype in ('float32', 'float64'):
            with self.subTest(dtype=dtype):
                x = np.full(1000, -1.0, dtype=dtype)
                parallel.random_fill(x)
                self.assertTrue(np.all(x >= 0))
                self.assertTrue(np.all(x < 1))

    def test_3d(self):
        for dtype in ('float32', 'float64'):
            with self.subTest(dtype=dtype):
                x = np.full((10, 10, 10), -1.0, dtype=dtype)
                parallel.random_fill(x)
                self.assertTrue(np.all(x >= 0))
                self.assertTrue(np.all(x < 1))
