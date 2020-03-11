import abc

from stencil_benchmarks.benchmark import Parameter, ParameterError

from stencil_benchmarks.benchmarks_collection.stencils import base

from .mixin import StencilMixin


class BasicStencilMixin(StencilMixin):
    loop = Parameter('loop kind', '1D', choices=['1D', '3D'])
    block_size = Parameter('block size', (1024, 1, 1))
    threads_per_block = Parameter(
        'threads per block (0 means equal to block size)', (0, 0, 0))

    @property
    def sorted_block_size(self):
        return self.sort_by_strides(self.block_size)

    @property
    def sorted_threads_per_block(self):
        return self.sort_by_strides(self.threads_per_block)

    def setup(self):
        if self.loop == '1D' and sum(b != 1 for b in self.block_size) != 1:
            raise ParameterError('block size must be 1 along '
                                 'all but one direction')
        if any(t > b for t, b in zip(self.threads_per_block, self.block_size)):
            raise ParameterError(
                'threads per block must be less than block size')
        self.threads_per_block = tuple(
            t if t > 0 else b
            for t, b in zip(self.threads_per_block, self.block_size))

        super().setup()

    @abc.abstractmethod
    def stencil_body(self):
        pass

    def template_file(self):
        return 'basic_' + self.loop.lower().replace("-", "_") + '.j2'

    def template_args(self):
        return dict(args=self.args,
                    ctype=self.ctype_name,
                    strides=self.strides,
                    sorted_strides=self.sorted_strides,
                    domain=self.domain,
                    sorted_domain=self.sorted_domain,
                    block_size=self.block_size,
                    sorted_block_size=self.sorted_block_size,
                    threads_per_block=self.threads_per_block,
                    sorted_threads_per_block=self.sorted_threads_per_block,
                    body=self.stencil_body(),
                    backend=self.backend,
                    gpu_timers=self.gpu_timers)


class Empty(BasicStencilMixin, base.EmptyStencil):
    def stencil_body(self):
        return ''


class Copy(BasicStencilMixin, base.CopyStencil):
    def stencil_body(self):
        return 'out[index] = inp[index];'


class OnesidedAverage(BasicStencilMixin, base.OnesidedAverageStencil):
    def stencil_body(self):
        stride = self.strides[self.axis]
        return (f'out[index] = (inp[index] + inp[index + {stride}]) /'
                f'{self.ctype_name}(2);')


class SymmetricAverage(BasicStencilMixin, base.SymmetricAverageStencil):
    def stencil_body(self):
        stride = self.strides[self.axis]
        return (f'out[index] = (inp[index - {stride}] + '
                f'inp[index + {stride}]) / {self.ctype_name}(2);')


class Laplacian(BasicStencilMixin, base.LaplacianStencil):
    def stencil_body(self):
        along_axes = (self.along_x, self.along_y, self.along_z)
        coeff = 2 * sum(along_axes)
        code = []
        for stride, apply_along_axis in zip(self.strides, along_axes):
            if apply_along_axis:
                code.append(f'inp[index - {stride}] + inp[index + {stride}]')

        return (f'out[index] = {self.ctype_name}({coeff}) * inp[index] - (' +
                ' + '.join(code) + ');')
