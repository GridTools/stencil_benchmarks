import unittest

from stencil_benchmarks.tools import alloc


class TestMalloc(unittest.TestCase):
    def test_malloc(self):
        size = 10000
        buffer = memoryview(alloc.malloc(size * 4)).cast('i')
        for i in range(size):
            buffer[i] = 42 * i

        for i in range(size):
            self.assertEqual(buffer[i], 42 * i)


class TestMmap(unittest.TestCase):
    def test_malloc(self):
        size = 10000
        buffer = memoryview(alloc.mmap(size * 4, False)).cast('i')
        for i in range(size):
            buffer[i] = 42 * i

        for i in range(size):
            self.assertEqual(buffer[i], 42 * i)
