from stencil_benchmarks.benchmark import Parameter
from stencil_benchmarks.benchmarks_collection.stencils import base

from .mixin import StencilMixin


class VerticalAdvectionMixin(StencilMixin):
    block_size = Parameter('block size', (32, 1))
    unroll_factor = Parameter(
        'force unrolling of vertical loop '
        '(-1: no enforced unrolling, 0: full unrolling,'
        ' > 0: unrolling factor)', -1)

    def template_file(self):
        return 'vertical_advection_' + type(self).__name__.lower() + '.j2'

    def template_args(self):
        return dict(**super().template_args(),
                    block_size=self.block_size,
                    unroll_factor=self.unroll_factor)


class Classic(VerticalAdvectionMixin, base.VerticalAdvectionStencil):
    pass


class SharedMem(VerticalAdvectionMixin, base.VerticalAdvectionStencil):
    pass


class LocalMem(VerticalAdvectionMixin, base.VerticalAdvectionStencil):
    pass


class LocalMemMerged(VerticalAdvectionMixin, base.VerticalAdvectionStencil):
    pass
