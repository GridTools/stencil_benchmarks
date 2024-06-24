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
import os
import re

import numpy as np

from ...benchmark import Benchmark, ExecutionError, Parameter, ParameterError
from ...tools import compilation, cpphelpers, template


class Original(Benchmark):
    array_size = Parameter('number of elements in arrays', 10000000)
    ntimes = Parameter('number of runs', 10)
    offset = Parameter('allocation offset', 0)
    dtype = Parameter('data type in NumPy format, e.g. float32 or float64',
                      'float64')
    compiler = Parameter('compiler path', 'gcc')
    compiler_flags = Parameter('compiler flags', '')
    platform_preset = Parameter('preset flags for specific hardware platform',
                                'native',
                                choices=['none', 'basic', 'native'])
    print_kernels = Parameter('print kernel code', False)

    def setup(self):
        super().setup()

        if self.compiler.endswith('icpc'):
            os.environ['KMP_INIT_AT_FORK'] = '0'

        template_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.template_file())
        code = template.render(template_file, **self.template_args())
        if self.print_kernels:
            print(
                re.search(
                    r'/\* [a-z ]*"tuned" versions of the kernels \*/(.*)',
                    cpphelpers.format_code(code),
                    re.MULTILINE | re.DOTALL).group(1))

        self.compiled = compilation.GnuLibrary(code,
                                               self.compile_command(),
                                               extension='.c')

    def compile_command(self):
        command = [self.compiler]
        if self.platform_preset != 'none':
            command += ['-Wall', '-DNDEBUG']
            if self.compiler.endswith('icc'):
                command += [
                    '-qopenmp', '-ffreestanding', '-O3', '-mcmodel=large',
                    '-qopt-streaming-stores=always'
                ]
            else:
                command += [
                    '-fopenmp', '-ffreestanding', '-Ofast', '-mcmodel=large'
                ]
            if self.platform_preset == 'native':
                command += ['-march=native', '-mtune=native']
        if self.compiler_flags:
            command += self.compiler_flags.split()
        return command

    def template_file(self):
        return 'original.j2'

    def template_args(self):
        return dict(array_size=self.array_size,
                    ntimes=self.ntimes,
                    offset=self.offset,
                    ctype=compilation.dtype_cname(self.dtype))

    def run(self):
        try:
            output = self.compiled.run()
        except compilation.ExecutionError as error:
            raise ExecutionError(*error.args) from error

        match = re.search(r'Failed Validation.*', output,
                          re.MULTILINE | re.DOTALL)
        if match:
            raise ExecutionError(match.group())

        ticks = int(re.search(r'= ([0-9]+) clock ticks', output).group(1))

        regex = re.compile(r'(Copy|Scale|Add|Triad): +'
                           r'([0-9.]+) +([0-9.]+) +'
                           r'([0-9.]+) +([0-9.]+)')
        results = []
        for match in regex.finditer(output):
            results.append({
                'name': match.group(1).lower(),
                'bandwidth': float(match.group(2)),
                'avg-time': float(match.group(3)),
                'time': float(match.group(4)),
                'max-time': float(match.group(5)),
                'ticks': ticks
            })

        return results


class _GccVecOps:
    def __init__(self, vector_size, dtype):
        itemsize = np.dtype(dtype).itemsize
        self.preamble = (
            f'typedef {compilation.dtype_cname(dtype)} vec_t '
            f'__attribute__((vector_size({vector_size * itemsize})));')
        self.vector_size = vector_size
        self.vector_type = 'vec_t'
        self.includes = []

    def load(self, ptr):
        return f'*({self.vector_type}*){ptr}'

    def store(self, ptr, value):
        return f'*({self.vector_type}*){ptr} = {value}'

    def broadcast(self, value):
        return f'({self.vector_type}){{' + ', '.join(
            str(value) for _ in range(self.vector_size)) + '}'

    def mul(self, a, b):
        return f'({a} * {b})'

    def add(self, a, b):
        return f'({a} + {b})'

    def fma(self, a, b, c):
        return f'({a} * {b} + {c})'


class _x86Ops:
    def __init__(self, ext, dtype, streaming_stores):
        itemsize = np.dtype(dtype).itemsize
        if ext == 'sse':
            self.vector_size = 16 // itemsize
            self._prefix = '_mm'
            self.vector_type = '__m128'
            self.includes = ['xmmintrin.h', 'emmintrin.h', 'immintrin.h']
        elif ext == 'avx':
            self.vector_size = 32 // itemsize
            self._prefix = '_mm256'
            self.vector_type = '__m256'
            self.includes = ['immintrin.h']
        elif ext == 'avx512':
            self.vector_size = 64 // itemsize
            self._prefix = '_mm512'
            self.vector_type = '__m512'
            self.includes = ['immintrin.h']
        else:
            raise ValueError('unsupported extension')
        if dtype == 'float32':
            self._suffix = 's'
        elif dtype == 'float64':
            self._suffix = 'd'
            self.vector_type += 'd'
        else:
            raise ValueError('unsupported dtype')
        self._streaming_stores = streaming_stores

    def load(self, ptr):
        return f'{self._prefix}_load_p{self._suffix}({ptr})'

    def store(self, ptr, value):
        if self._streaming_stores:
            return f'{self._prefix}_stream_p{self._suffix}({ptr}, {value})'
        return f'{self._prefix}_store_p{self._suffix}({ptr}, {value})'

    def broadcast(self, value):
        return f'{self._prefix}_set1_p{self._suffix}({value})'

    def mul(self, a, b):
        return f'{self._prefix}_mul_p{self._suffix}({a}, {b})'

    def add(self, a, b):
        return f'{self._prefix}_add_p{self._suffix}({a}, {b})'

    def fma(self, a, b, c):
        return f'{self._prefix}_fmadd_p{self._suffix}({a}, {b}, {c})'


class Native(Original):
    architecture = Parameter(
        'hardware architecture',
        'gcc-vec',
        choices=['gcc-vec', 'x86-sse', 'x86-avx', 'x86-avx512'])
    unroll_factor = Parameter('loop unroll factor', 1)
    fma = Parameter('use fused multiply-add instructions', True)
    vector_size = Parameter('vector size (only where applicable)', 0)
    streaming_stores = Parameter('bypass load of destination array', True)

    def template_file(self):
        return 'native.j2'

    def template_args(self):
        if self.architecture == 'gcc-vec':
            ops = _GccVecOps(max(self.vector_size, 1), self.dtype)
        elif self.architecture.startswith('x86'):
            ops = _x86Ops(
                self.architecture.split('-')[1], self.dtype,
                self.streaming_stores)
        else:
            raise ParameterError(
                f'unsupported architecture {self.architecture}')
        return dict(**super().template_args(),
                    unroll_factor=self.unroll_factor,
                    ops=ops)
