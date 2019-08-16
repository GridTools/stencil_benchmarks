import abc
import os
import warnings

import numpy as np

from .... import benchmark
from ....tools import cpphelpers, compilation, template

# pylint: disable=abstract-method,invalid-sequence-index,no-member


class StencilMixin(benchmark.Benchmark):
    compiler = benchmark.Parameter('compiler path', 'g++')
    compiler_flags = benchmark.Parameter('compiler flags', '')
    platform_preset = benchmark.Parameter(
        'preset flags for specific hardware platform',
        'native',
        choices=['none', 'native', 'knl'])
    print_code = benchmark.Parameter('print generated code', False)

    def setup(self):
        super().setup()

        template_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.template_file())
        code = template.render(template_file, **self.template_args())

        if self.print_code:
            print(cpphelpers.format_code(code))

        if self.compiler.endswith('icpc'):
            os.environ['KMP_INIT_AT_FORK'] = '0'

        compile_command = [self.compiler]
        if self.platform_preset != 'none':
            compile_command += ['-xc++', '-std=c++11', '-Wall', '-DNDEBUG']
            if self.compiler.endswith('icpc'):
                compile_command += ['-qopenmp', '-ffreestanding', '-O3']
            else:
                compile_command += ['-fopenmp', '-Ofast']

            if self.platform_preset == 'native':
                compile_command += ['-march=native', '-mtune=native', '-Ofast']
            elif self.platform_preset == 'knl':
                if not self.huge_pages:
                    warnings.warn('enabling huge pages on KNL is recommended '
                                  '(use --huge-pages to enable)')
                if not self.offset_allocations:
                    warnings.warn(
                        'offsetting allocations on KNL is recommended '
                        '(use --offset-allocations to enable)')
                if self.compiler.endswith('icpc'):
                    compile_command += ['-xmic-avx512']
                else:
                    compile_command += [
                        '-march=knl', '-mtune=knl',
                        '-fvect-cost-model=unlimited'
                    ]
        compile_command += self.compiler_flags.split()

        self.compiled = compilation.gnu_func(compile_command, code, 'kernel',
                                             float)

    @property
    def ctype_name(self):
        return compilation.ctype_cname(compilation.dtype_as_ctype(self.dtype))

    @property
    def sorted_domain(self):
        # pylint: disable=invalid-unary-operand-type
        indices = np.argsort(-np.array(self.strides))
        return tuple(np.array(self.domain)[indices])

    @property
    def sorted_strides(self):
        return tuple(sorted(self.strides, key=lambda x: -x))

    @property
    def sorted_block_size(self):
        # pylint: disable=invalid-unary-operand-type
        indices = np.argsort(-np.array(self.strides))
        return tuple(np.array(self.block_size)[indices])

    @abc.abstractmethod
    def template_file(self):
        pass

    @abc.abstractmethod
    def template_args(self):
        pass

    def run_stencil(self, data):
        offset = (self.halo, ) * 3
        return self.compiled(*(compilation.data_ptr(array, offset)
                               for array in data))


class BasicStencilMixin(StencilMixin):
    loop = benchmark.Parameter('loop kind',
                               '1D',
                               choices=['1D', '3D', '3D-blocked'])
    block_size = benchmark.Parameter('block_size', (1, 1, 1))

    @abc.abstractmethod
    def stencil_body(self):
        pass

    def template_file(self):
        return 'basic_' + self.loop.lower().replace("-", "_") + '.template'

    def template_args(self):
        return dict(args=self.args,
                    ctype=self.ctype_name,
                    strides=self.strides,
                    sorted_strides=self.sorted_strides,
                    domain=self.domain,
                    sorted_domain=self.sorted_domain,
                    block_size=self.block_size,
                    sorted_block_size=self.sorted_block_size,
                    body=self.stencil_body())


class VerticalAdvectionMixin(StencilMixin):
    block_size = benchmark.Parameter('block size', (1, 1))

    def template_file(self):
        return ('vertical_advection_' + type(self).__name__.lower() +
                '.template')

    def template_args(self):
        return dict(args=self.args,
                    ctype=self.ctype_name,
                    strides=self.strides,
                    domain=self.domain,
                    block_size=self.block_size)


class HorizontalDiffusionMixin(StencilMixin):
    block_size = benchmark.Parameter('block size', (1, 1, 1))

    def template_file(self):
        return ('horizontal_diffusion_' + type(self).__name__.lower() +
                '.template')

    def template_args(self):
        return dict(args=self.args,
                    ctype=self.ctype_name,
                    strides=self.strides,
                    domain=self.domain,
                    block_size=self.block_size)
