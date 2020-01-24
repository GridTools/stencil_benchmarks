import unittest

import numpy as np

from stencil_benchmarks.tools import array


class TestAllocArray(unittest.TestCase):
    def test_basic(self):
        data = array.alloc_array((3, 5, 7), 'int32', (0, 2, 1))
        self.assertEqual(data.shape, (3, 5, 7))
        self.assertEqual(data.dtype, np.dtype('int32'))
        self.assertTrue(data.strides[1] < data.strides[2] < data.strides[0])
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    data[i, j, k] = i + 2 * j + 6 * k
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    self.assertEqual(data[i, j, k], i + 2 * j + 6 * k)

    def test_alignment(self):
        data = array.alloc_array((3, 5, 7), 'int32', (0, 2, 1), alignment=64)
        self.assertEqual(data.shape, (3, 5, 7))
        self.assertEqual(data.dtype, np.dtype('int32'))
        self.assertTrue(data.strides[1] < data.strides[2] < data.strides[0])
        self.assertTrue(data.ctypes.data % 64 == 0)
        self.assertTrue(data.strides[2] % 64 == 0)
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    data[i, j, k] = i + 2 * j + 6 * k
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    self.assertEqual(data[i, j, k], i + 2 * j + 6 * k)

    def test_index_to_align(self):
        data = array.alloc_array((3, 5, 7),
                                 'int32', (0, 2, 1),
                                 alignment=64,
                                 index_to_align=(1, 1, 2))
        self.assertEqual(data.shape, (3, 5, 7))
        self.assertEqual(data.dtype, np.dtype('int32'))
        self.assertTrue(data.strides[1] < data.strides[2] < data.strides[0])
        self.assertTrue(data[1:, 1:, 2:].ctypes.data % 64 == 0)
        self.assertTrue(data.strides[2] % 64 == 0)
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    data[i, j, k] = i + 2 * j + 6 * k
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    self.assertEqual(data[i, j, k], i + 2 * j + 6 * k)
