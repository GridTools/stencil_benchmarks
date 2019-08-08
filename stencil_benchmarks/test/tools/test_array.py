import ctypes
import unittest

import numpy as np

from stencil_benchmarks.tools import array


class TestAllocBuffer(unittest.TestCase):
    def test_alloc_free(self):
        address = 0
        allocated = set()

        def mock_alloc(nbytes):
            nonlocal address
            fake_pointer = address
            allocated.add(fake_pointer)
            address += nbytes
            return fake_pointer

        def mock_free(pointer):
            allocated.remove(pointer)

        buffers = []
        buffers.append(array.alloc_buffer(128, mock_alloc, mock_free))
        self.assertEqual(len(allocated), 1)
        self.assertEqual(ctypes.addressof(buffers[0]), 0)

        buffers.append(array.alloc_buffer(42, mock_alloc, mock_free))
        self.assertEqual(len(allocated), 2)
        self.assertEqual(ctypes.addressof(buffers[1]), 128)
        buffers.append(array.alloc_buffer(42, mock_alloc, mock_free))
        self.assertEqual(len(allocated), 3)
        self.assertEqual(ctypes.addressof(buffers[2]), 128 + 42)

        del buffers[1]
        self.assertEqual(len(allocated), 2)
        self.assertEqual(ctypes.addressof(buffers[0]), 0)
        self.assertEqual(ctypes.addressof(buffers[1]), 128 + 42)

        buffers.clear()
        self.assertFalse(allocated)


class TestCmallocCfree(unittest.TestCase):
    def test_alloc_free(self):
        pointer = array.cmalloc(16)
        buffer = (ctypes.c_int32 * 4).from_address(pointer)
        for i in range(4):
            buffer[i] = 42 * i

        for i in range(4):
            self.assertEqual(buffer[i], 42 * i)

        array.cfree(pointer)


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
