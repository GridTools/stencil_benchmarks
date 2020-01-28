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


class TestCacheFunctions(unittest.TestCase):
    def test_l1_dcache_assoc(self):
        self.assertGreaterEqual(alloc.l1_dcache_assoc(), 1)

    def test_l1_dcache_linesize(self):
        self.assertGreaterEqual(alloc.l1_dcache_linesize(), 4)

    def test_l1_dcache_size(self):
        self.assertGreaterEqual(alloc.l1_dcache_size(), 0)
