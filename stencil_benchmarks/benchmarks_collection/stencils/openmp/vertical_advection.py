from .mixin import VerticalAdvectionMixin
from ..base import VerticalAdvectionStencil
from .... import benchmark


class KInnermost(VerticalAdvectionMixin, VerticalAdvectionStencil):
    prefetch_distance = benchmark.Parameter('vertical prefetching distance', 0)

    def template_args(self):
        return dict(**super().template_args(),
                    prefetch_distance=self.prefetch_distance)


class KMiddle(VerticalAdvectionMixin, VerticalAdvectionStencil):
    pass
