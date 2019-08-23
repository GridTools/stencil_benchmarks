from .mixin import BasicStencilMixin
from ..base import (EmptyStencil, CopyStencil, OnesidedAverageStencil,
                    SymmetricAverageStencil, LaplacianStencil)

# pylint: disable=invalid-sequence-index


class Empty(BasicStencilMixin, EmptyStencil):
    def stencil_body(self):
        return ''


class Copy(BasicStencilMixin, CopyStencil):
    def stencil_body(self):
        return 'out[index] = inp[index];'


class OnesidedAverage(BasicStencilMixin, OnesidedAverageStencil):
    def stencil_body(self):
        stride = self.strides[self.axis]
        return f'out[index] = (inp[index] + inp[index + {stride}]) / 2;'


class SymmetricAverage(BasicStencilMixin, SymmetricAverageStencil):
    def stencil_body(self):
        stride = self.strides[self.axis]
        return (f'out[index] = (inp[index - {stride}] + '
                f'inp[index + {stride}]) / 2;')


class Laplacian(BasicStencilMixin, LaplacianStencil):
    def stencil_body(self):
        along_axes = (self.along_x, self.along_y, self.along_z)
        coeff = 2 * sum(along_axes)
        code = []
        for stride, apply_along_axis in zip(self.strides, along_axes):
            if apply_along_axis:
                code.append(f'inp[index - {stride}] + inp[index + {stride}]')

        return f'out[index] = {coeff} * inp[index] - (' + ' + '.join(
            code) + ');'
