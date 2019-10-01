from .mixin import VerticalAdvectionMixin
from ..base import VerticalAdvectionStencil


class Classic(VerticalAdvectionMixin, VerticalAdvectionStencil):
    pass


class SharedMem(VerticalAdvectionMixin, VerticalAdvectionStencil):
    pass


class LocalMem(VerticalAdvectionMixin, VerticalAdvectionStencil):
    pass


class LocalMemMerged(VerticalAdvectionMixin, VerticalAdvectionStencil):
    pass
