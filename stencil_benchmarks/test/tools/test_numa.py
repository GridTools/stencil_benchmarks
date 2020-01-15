import ctypes
import unittest

from stencil_benchmarks.tools import numa


class TestAllNodes(unittest.TestCase):
    def test_at_least_one_node(self):
        self.assertGreater(len(numa.all_nodes()), 0)


class TestAllocFree(unittest.TestCase):
    def test_alloc_free(self):
        for node in numa.all_nodes():
            size = 10000
            pointer = numa.alloc_onnode(size * 4, node)
            buffer = (ctypes.c_int32 * size).from_address(pointer)
            for i in range(size):
                buffer[i] = 42 * i

            for i in range(size):
                self.assertEqual(buffer[i], 42 * i)

            numa.free(pointer, size * 4)
