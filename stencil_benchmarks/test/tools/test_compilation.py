import ctypes
import unittest

import numpy as np

from stencil_benchmarks.tools import compilation


class TestGnuLibrary(unittest.TestCase):
    def test_c(self):
        code = '''
            int func(int i) {
                return 2 * i;
            }'''

        library = compilation.gnu_library(['gcc', '-xc'], code)
        self.assertEqual(library.func(42), 84)

    def test_cpp(self):
        code = '''
            extern "C" int func(int i) {
                return 2 * i;
            }'''

        library = compilation.gnu_library(['g++', '-xc++'], code)
        self.assertEqual(library.func(42), 84)


class TestGnuFunc(unittest.TestCase):
    def test_c(self):
        code = '''
            int func(int i) {
                return 2 * i;
            }'''

        func = compilation.gnu_func(['gcc', '-xc'], code, 'func')
        self.assertEqual(func(42), 84)

    def test_cpp(self):
        code = '''
            extern "C" int func(int i) {
                return 2 * i;
            }'''

        func = compilation.gnu_func(['g++', '-xc++'], code, 'func')
        self.assertEqual(func(42), 84)


class TestDtypeAsCtype(unittest.TestCase):
    def check_type(self, dtype, ctype):
        self.assertIs(compilation.dtype_as_ctype(np.dtype(dtype)), ctype)

    def test_floats(self):
        self.check_type('float32', ctypes.c_float)
        self.check_type('float64', ctypes.c_double)

    def test_signed_ints(self):
        self.check_type('int8', ctypes.c_int8)
        self.check_type('int16', ctypes.c_int16)
        self.check_type('int32', ctypes.c_int32)
        self.check_type('int64', ctypes.c_int64)

    def test_unsigned_ints(self):
        self.check_type('uint8', ctypes.c_uint8)
        self.check_type('uint16', ctypes.c_uint16)
        self.check_type('uint32', ctypes.c_uint32)
        self.check_type('uint64', ctypes.c_uint64)


class TestCtypeCnam(unittest.TestCase):
    def check_type(self, ctype, name):
        self.assertEqual(compilation.ctype_cname(ctype), name)

    def test_floats(self):
        self.check_type(ctypes.c_float, 'float')
        self.check_type(ctypes.c_double, 'double')

    def test_signed_ints(self):
        self.check_type(ctypes.c_int8, 'std::int8_t')
        self.check_type(ctypes.c_int16, 'std::int16_t')
        self.check_type(ctypes.c_int32, 'std::int32_t')
        self.check_type(ctypes.c_int64, 'std::int64_t')

    def test_unsigned_ints(self):
        self.check_type(ctypes.c_uint8, 'std::uint8_t')
        self.check_type(ctypes.c_uint16, 'std::uint16_t')
        self.check_type(ctypes.c_uint32, 'std::uint32_t')
        self.check_type(ctypes.c_uint64, 'std::uint64_t')


class TestDataPtr(unittest.TestCase):
    def test_access(self):
        array = np.random.uniform(size=10).astype('float32')
        ptr = compilation.data_ptr(array)
        for i in range(array.size):
            self.assertEqual(ptr[i], array[i])

    def test_int_offset(self):
        array = np.arange(7)
        for i in range(array.size):
            ptr = compilation.data_ptr(array, offset=i)
            self.assertEqual(ptr[0], array[i])

    def test_tuple_offset(self):
        array = np.random.uniform(size=(7, 3))
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                ptr = compilation.data_ptr(array, offset=(i, j))
                self.assertEqual(ptr[0], array[i, j])
