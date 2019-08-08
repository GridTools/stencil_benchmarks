import abc
import os

import numpy as np

from .. import base
from .... import benchmark
from ....tools import cpphelpers, compilation, template

# pylint: disable=abstract-method,invalid-sequence-index,no-member


class StencilMixin(benchmark.Benchmark):
    compiler = benchmark.Parameter('compiler path', str, default='g++')
    compiler_flags = benchmark.Parameter(
        'compiler flags',
        str,
        default='-xc++ -Ofast -fopenmp -Wall -march=native')
    print_code = benchmark.Parameter('print generated code',
                                     bool,
                                     default=False)

    def setup(self):
        super().setup()

        offset = (self.halo, ) * 3
        self.inout_ptrs = [[
            compilation.data_ptr(array, offset) for array in inout
        ] for inout in self.inouts]

        code = self.generate_code()

        if self.print_code:
            print(cpphelpers.format_code(code))

        self.compiled = compilation.gnu_func(
            [self.compiler] + self.compiler_flags.split(), code, 'kernel')

    @property
    def ctype_name(self):
        return compilation.ctype_cname(compilation.dtype_as_ctype(self.dtype))

    @property
    def strides(self):
        return tuple(
            np.array(self.inouts[0][0].strides) // self.dtype.itemsize)

    @property
    @abc.abstractproperty
    def args(self):
        pass

    @abc.abstractmethod
    def generate_code(self):
        pass

    @abc.abstractmethod
    def stencil_body(self):
        pass

    def run_stencil(self, data_set):
        self.compiled(*self.inout_ptrs[data_set])


class BasicStencilMixin(StencilMixin):
    loop = benchmark.Parameter('loop kind', str, default='1d')
    block_size = benchmark.Parameter('block_size',
                                     int,
                                     nargs=3,
                                     default=(1, 1, 1))

    def generate_code(self):
        template_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f'loop_{self.loop.lower().replace("-", "_")}.template')
        return template.render(template_file,
                               args=self.args,
                               ctype=self.ctype_name,
                               strides=self.strides,
                               domain=self.domain,
                               block_size=self.block_size,
                               body=self.stencil_body())

    @property
    def args(self):
        return 'inp', 'out'


class Copy(BasicStencilMixin, base.CopyStencil):
    def stencil_body(self):
        return 'out[index] = inp[index];'


class OnesidedAverage(BasicStencilMixin, base.OnesidedAverageStencil):
    def stencil_body(self):
        stride = self.strides[self.axis]
        return f'out[index] = (inp[index] + inp[index + {stride}]) / 2;'


class SymmetricAverage(BasicStencilMixin, base.SymmetricAverageStencil):
    def stencil_body(self):
        stride = self.strides[self.axis]
        return (f'out[index] = (inp[index - {stride}] + '
                f'inp[index + {stride}]) / 2;')


class Laplacian(BasicStencilMixin, base.LaplacianStencil):
    def stencil_body(self):
        along_axes = (self.along_x, self.along_y, self.along_z)
        coeff = 2 * sum(along_axes)
        code = []
        for stride, apply_along_axis in zip(self.strides, along_axes):
            if apply_along_axis:
                code.append(f'inp[index - {stride}] + inp[index + {stride}]')

        return f'out[index] = {coeff} * inp[index] - (' + ' + '.join(
            code) + ');'


class HorizontalDiffusionOtf(BasicStencilMixin,
                             base.HorizontalDiffusionStencil):
    @property
    def args(self):
        return 'inp', 'coeff', 'out'

    def stencil_body(self):
        stride_x, stride_y, _ = self.strides
        return f'''const auto inp_ij = inp[index];
                const auto inp_im1j = inp[index - {stride_x}];
                const auto inp_ip1j = inp[index + {stride_x}];
                const auto inp_ijm1 = inp[index - {stride_y}];
                const auto inp_ijp1 = inp[index + {stride_y}];
                const auto inp_im2j = inp[index - 2 * {stride_x}];
                const auto inp_im1jm1 = inp[index - {stride_x} - {stride_y}];
                const auto inp_im1jp1 = inp[index - {stride_x} + {stride_y}];
                const auto inp_ip2j = inp[index + 2 * {stride_x}];
                const auto inp_ip1jm1 = inp[index + {stride_x} - {stride_y}];
                const auto inp_ip1jp1 = inp[index + {stride_x} + {stride_y}];
                const auto inp_ijm2 = inp[index - 2 * {stride_y}];
                const auto inp_ijp2 = inp[index + 2 * {stride_y}];

                const auto lap_ij = 4 * inp_ij - inp_im1j - inp_ip1j -
                                inp_ijm1 - inp_ijp1;
                const auto lap_imj = 4 * inp_im1j - inp_im2j -
                                    inp_ij - inp_im1jm1 -
                                    inp_im1jp1;
                const auto lap_ipj = 4 * inp_ip1j - inp_ij -
                                    inp_ip2j - inp_ip1jm1 -
                                    inp_ip1jp1;
                const auto lap_ijm = 4 * inp_ijm1 - inp_im1jm1 -
                                    inp_ip1jm1 - inp_ijm2 -
                                    inp_ij;
                const auto lap_ijp = 4 * inp_ijp1 - inp_im1jp1 -
                                    inp_ip1jp1 - inp_ij -
                                    inp_ijp2;

                auto flx_ij = lap_ipj - lap_ij;
                flx_ij = flx_ij * (inp_ip1j - inp_ij) > 0 ? 0 : flx_ij;

                auto flx_imj = lap_ij - lap_imj;
                flx_imj = flx_imj * (inp_ij - inp_im1j) > 0 ? 0 : flx_imj;

                auto fly_ij = lap_ijp - lap_ij;
                fly_ij = fly_ij * (inp_ijp1 - inp_ij) > 0 ? 0 : fly_ij;

                auto fly_ijm = lap_ij - lap_ijm;
                fly_ijm = fly_ijm * (inp_ij - inp_ijm1) > 0 ? 0 : fly_ijm;

                out[index] = inp_ij - coeff[index] * (flx_ij - flx_imj +
                                                      fly_ij - fly_ijm);
                '''
