# Stencil Benchmarks
#
# Copyright (c) 2017-2021, ETH Zurich and MeteoSwiss
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
