import abc
import ctypes
import os
import warnings

import numpy as np

from stencil_benchmarks.benchmark import Benchmark, ExecutionError, Parameter
from stencil_benchmarks.tools import cpphelpers, compilation, template


class StencilMixin(Benchmark):
    compiler = Parameter('compiler path', 'g++')
    compiler_flags = Parameter('compiler flags', '')
    platform_preset = Parameter('preset flags for specific hardware platform',
                                'native',
                                choices=['none', 'native', 'knl'])
    print_code = Parameter('print generated code', False)
    numa = Parameter('enable NUMA-awareness', False)
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
            elif self.platform_preset == 'knl':
                if not self.huge_pages:
                    warnings.warn('enabling huge pages on KNL is recommended '
                                  '(use --huge-pages to enable)')
                if not self.offset_allocations:
                    warnings.warn(
                        'offsetting allocations on KNL is recommended '
                        '(use --offset-allocations to enable)')
                if self.compiler.endswith('icpc'):
                    command += ['-xmic-avx512']
                else:
                    command += [
                        '-march=knl', '-mtune=knl',
                        '-fvect-cost-model=unlimited'
                    ]
        if self.compiler_flags:
            command += self.compiler_flags.split()
        if self.numa:
            command += ['-lnuma']
        return command

    @property
    def ctype_name(self):
        return compilation.dtype_cname(self.dtype)

    @property
    def sorted_domain(self):
        # pylint: disable=invalid-unary-operand-type
        indices = np.argsort(-np.array(self.strides))
        return tuple(np.array(self.domain)[indices])

    @property
    def sorted_strides(self):
        return tuple(sorted(self.strides, key=lambda x: -x))

    @abc.abstractmethod
    def template_file(self):
        pass

    def template_args(self):
        return dict(args=self.args,
                    ctype=self.ctype_name,
                    strides=self.strides,
                    sorted_strides=self.sorted_strides,
                    domain=self.domain,
                    sorted_domain=self.sorted_domain,
                    numa=self.numa,
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
