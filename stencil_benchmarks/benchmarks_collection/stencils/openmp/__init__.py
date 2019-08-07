import abc
import os

import numpy as np

from .. import base
from .... import benchmark
from ....tools import compilation, template

# pylint: disable=invalid-sequence-index,no-member


class StencilMixin(benchmark.Benchmark):
    compiler = benchmark.Parameter('compiler path', str, default='g++')
    compiler_flags = benchmark.Parameter(
        'compiler flags',
        str,
        default='-xc++ -Ofast -fopenmp -Wall -march=native')
    loop = benchmark.Parameter('loop kind', str, default='1d')

    def setup(self):
        super().setup()

        offset = (self.halo, ) * 3
        self.inout_ptrs = [(compilation.data_ptr(inp, offset),
                            compilation.data_ptr(out, offset))
                           for inp, out in self.inouts]

        code = template.render(self.template_file,
                               ctype=self.ctype_name,
                               strides=self.strides,
                               domain=self.domain,
                               body=self.stencil_body())

        self.compiled = compilation.gnu_func(
            [self.compiler] + self.compiler_flags.split(), code, 'kernel')

    @property
    def data_size(self):
        return 2 * np.product(self.domain) * self.dtype.itemsize

    @property
    def ctype_name(self):
        return compilation.ctype_cname(compilation.dtype_as_ctype(self.dtype))

    @property
    def strides(self):
        return tuple(
            np.array(self.inouts[0][0].strides) // self.dtype.itemsize)

    @property
    def template_file(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f'loop_{self.loop.lower()}.template')

    @abc.abstractmethod
    def stencil_body(self):
        pass

    def run_stencil(self, data_set):
        self.compiled(*self.inout_ptrs[data_set])


class Copy(StencilMixin, base.CopyStencil):
    def stencil_body(self):
        return 'out[index] = inp[index];'


class OnesidedAverage(StencilMixin, base.OnesidedAverageStencil):
    def stencil_body(self):
        stride = self.strides[self.axis]
        return f'out[index] = (inp[index] + inp[index + {stride}]) / 2;'


class SymmetricAverage(StencilMixin, base.SymmetricAverageStencil):
    def stencil_body(self):
        stride = self.strides[self.axis]
        return (f'out[index] = (inp[index - {stride}] + '
                f'inp[index + {stride}]) / 2;')


class Laplacian(StencilMixin, base.LaplacianStencil):
    def stencil_body(self):
        along_axes = (self.along_x, self.along_y, self.along_z)
        coeff = 2 * sum(along_axes)
        code = []
        for stride, apply_along_axis in zip(self.strides, along_axes):
            if apply_along_axis:
                code.append(f'inp[index - {stride}] + inp[index + {stride}]')

        return f'out[index] = {coeff} * inp[index] - (' + ' + '.join(
            code) + ');'
