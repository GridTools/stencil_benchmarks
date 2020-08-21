# Stencil Benchmarks
#
# Copyright (c) 2017-2020, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# SPDX-License-Identifier: BSD-3-Clause
import ctypes
import unittest

import numpy as np

from stencil_benchmarks.tools import compilation


class TestGnuLibrary(unittest.TestCase):
    def test_c(self):
        code = '''
            int func(int *i) {
                *i *= 2;
                return 0;
            }'''

        library = compilation.GnuLibrary(code, extension='.c')
        ival = ctypes.c_int(42)
        library.func(ctypes.byref(ival))
        self.assertEqual(ival.value, 84)

    def test_cpp(self):
        code = '''
            extern "C" int func(float *f) {
                *f *= 2;
                return 0;
            }
            '''

        library = compilation.GnuLibrary(code)
        fval = ctypes.c_float(42)
        library.func(ctypes.byref(fval))
        self.assertEqual(fval.value, 84.0)

    def test_output_capture(self):
        code = '''
            #include <iostream>

            extern "C" int foo(float f, int i) {
                std::cout << (f * i) << std::endl;
                return 0;
            }
            '''

        library = compilation.GnuLibrary(code)
        self.assertEqual(
            library.foo(0.5, 42, argtypes=[ctypes.c_float, ctypes.c_int]),
            '21\n')
        self.assertEqual(
            library.foo(2.0, 42, argtypes=[ctypes.c_float, ctypes.c_int]),
            '84\n')

    def test_compilation_error(self):
        code = '''
            extern "C" int foo(float *f, int i) {
                return 0 // missing semicolon
            }
            '''
        with self.assertRaises(compilation.CompilationError):
            compilation.GnuLibrary(code)

    def test_execution_error(self):
        code = '''
            #include <iostream>

            extern "C" int foo(float *f, int i) {
                if (!f) {
                    std::cerr << "nullpointer!" << std::endl;
                    return 1;
                }
                *f *= i;
                return 0;
            }
            '''

        library = compilation.GnuLibrary(code)
        fval = ctypes.c_float(0.5)
        self.assertEqual(library.foo(ctypes.byref(fval), 42), '')
        self.assertEqual(fval.value, 21.0)

        with self.assertRaisesRegex(compilation.ExecutionError,
                                    'nullpointer!\n'):
            library.foo(ctypes.c_void_p(), 42)


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


class TestCtypeCname(unittest.TestCase):
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


class TestDtypeCname(unittest.TestCase):
    def check_type(self, dtype, name):
        self.assertEqual(compilation.dtype_cname(np.dtype(dtype)), name)

    def test_floats(self):
        self.check_type('float16', 'half')
        self.check_type('float32', 'float')
        self.check_type('float64', 'double')

    def test_signed_ints(self):
        self.check_type('int8', 'std::int8_t')
        self.check_type('int16', 'std::int16_t')
        self.check_type('int32', 'std::int32_t')
        self.check_type('int64', 'std::int64_t')

    def test_unsigned_ints(self):
        self.check_type('uint8', 'std::uint8_t')
        self.check_type('uint16', 'std::uint16_t')
        self.check_type('uint32', 'std::uint32_t')
        self.check_type('uint64', 'std::uint64_t')


class TestDataPtr(unittest.TestCase):
    def test_access(self):
        array = np.random.uniform(size=10).astype('float32')
        ptr = ctypes.cast(compilation.data_ptr(array),
                          ctypes.POINTER(ctypes.c_float))
        for i in range(array.size):
            self.assertEqual(ptr[i], array[i])

    def test_int_offset(self):
        array = np.arange(7)
        for i in range(array.size):
            ptr = ctypes.cast(compilation.data_ptr(array, offset=i),
                              ctypes.POINTER(ctypes.c_int64))
            self.assertEqual(ptr[0], array[i])

    def test_tuple_offset(self):
        array = np.random.uniform(size=(7, 3))
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                ptr = ctypes.cast(compilation.data_ptr(array, offset=(i, j)),
                                  ctypes.POINTER(ctypes.c_double))
                self.assertEqual(ptr[0], array[i, j])
