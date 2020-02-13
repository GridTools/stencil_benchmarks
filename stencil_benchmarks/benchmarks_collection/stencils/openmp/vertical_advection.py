from .mixin import VerticalAdvectionMixin
from ..base import VerticalAdvectionStencil
from .... import benchmark


class KInnermost(VerticalAdvectionMixin, VerticalAdvectionStencil):
    prefetch_distance = benchmark.Parameter('vertical prefetching distance', 0)

    def template_args(self):
        return dict(**super().template_args(),
                    prefetch_distance=self.prefetch_distance)


class KInnermostVec(KInnermost):
    vector_size = benchmark.Parameter('vector size', 16)

    def template_args(self):
        return dict(**super().template_args(), vector_size=self.vector_size)


class KInnermostBlockVec(KInnermostVec):
    def setup(self):
        super().setup()
        if self.block_size[0] % self.vector_size != 0:
            raise benchmark.ParameterError(
                'block size must be divisible by vector size')


class KMiddle(VerticalAdvectionMixin, VerticalAdvectionStencil):
    pass
