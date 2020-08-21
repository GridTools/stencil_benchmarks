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
import abc
import ctypes
import os

from stencil_benchmarks.benchmark import Benchmark, ExecutionError, Parameter
from stencil_benchmarks.tools import cpphelpers, compilation, template


class StencilMixin(Benchmark):
    compiler = Parameter('compiler path', 'g++')
    compiler_flags = Parameter('compiler flags', '')
    platform_preset = Parameter('preset flags for specific hardware platform',
                                'native',
                                choices=['none', 'native'])
    print_code = Parameter('print generated code', False)
    streaming_stores = Parameter(
        'enable streaming/non-temporal store instructions '
        '(only supported by some implementations)', False)

    def setup(self):
        super().setup()

        template_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'templates',
            self.template_file())
        code = template.render(template_file, **self.template_args())

        if self.print_code:
            print(cpphelpers.format_code(code))

        if self.compiler.endswith('icpc'):
            os.environ['KMP_INIT_AT_FORK'] = '0'

        self.compiled = compilation.GnuLibrary(code, self.compile_command())

    def compile_command(self):
        command = [self.compiler]
        if self.platform_preset != 'none':
            command += ['-std=c++11', '-Wall', '-DNDEBUG']
            if self.compiler.endswith('icpc'):
                command += ['-qopenmp', '-ffreestanding', '-O3']
            else:
                command += ['-fopenmp', '-Ofast']

            if self.platform_preset == 'native':
                command += ['-march=native', '-mtune=native', '-Ofast']
        if self.compiler_flags:
            command += self.compiler_flags.split()
        return command

    @abc.abstractmethod
    def template_file(self):
        pass

    def template_args(self):
        return dict(args=self.args,
                    ctype=compilation.dtype_cname(self.dtype),
                    strides=self.strides,
                    domain=self.domain,
                    halo=self.halo,
                    alignment=self.alignment,
                    streaming_stores=self.streaming_stores)

    def run_stencil(self, data):
        offset = (self.halo, ) * 3
        time = ctypes.c_double()
        try:
            self.compiled.kernel(
                ctypes.byref(time),
                *(compilation.data_ptr(array, offset) for array in data))
        except compilation.ExecutionError as error:
            raise ExecutionError() from error
        return time.value
