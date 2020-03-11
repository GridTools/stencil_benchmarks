from stencil_benchmarks.benchmark import Parameter, ParameterError
from stencil_benchmarks.benchmarks_collection.stencils import base

from .mixin import StencilMixin


class VerticalAdvectionMixin(StencilMixin):
    block_size = Parameter('block size', (8, 1))

    def template_file(self):
        return 'vertical_advection_' + type(self).__name__.lower() + '.j2'

    def template_args(self):
        return dict(**super().template_args(), block_size=self.block_size)


class KInnermost(VerticalAdvectionMixin, base.VerticalAdvectionStencil):
    prefetch_distance = Parameter('vertical prefetching distance', 0)

    def template_args(self):
        return dict(**super().template_args(),
                    prefetch_distance=self.prefetch_distance)


class KInnermostVec(KInnermost):
    vector_size = Parameter('vector size', 8)

    def template_args(self):
        return dict(**super().template_args(), vector_size=self.vector_size)


class KInnermostBlockVec(KInnermostVec):
    def setup(self):
        super().setup()
        if self.block_size[0] % self.vector_size != 0:
            raise ParameterError('block size must be divisible by vector size')


class KMiddle(VerticalAdvectionMixin, base.VerticalAdvectionStencil):
    pass


class KMiddleVec(KMiddle):
    vector_size = Parameter('vector size', 8)

    def template_args(self):
        return dict(**super().template_args(), vector_size=self.vector_size)
