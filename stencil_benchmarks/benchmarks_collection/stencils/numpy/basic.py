import numpy as np

from ..base import (CopyStencil, OnesidedAverageStencil,
                    SymmetricAverageStencil, LaplacianStencil)
from ....tools import timing


class Copy1D(CopyStencil):
    @timing.return_time
    def run_stencil(self, data):
        data.out[:] = data.inp[:]


class Copy(CopyStencil):
    @timing.return_time
    def run_stencil(self, data):
        data.out[self.inner_slice()] = data.inp[self.inner_slice()]


class OnesidedAverage(OnesidedAverageStencil):
    @timing.return_time
    def run_stencil(self, data):
        shift = np.zeros(3, dtype=int)
        shift[self.axis] = 1
        data.out[self.inner_slice()] = (data.inp[self.inner_slice(shift)] +
                                        data.inp[self.inner_slice()]) / 2


class SymmetricAverage(SymmetricAverageStencil):
    @timing.return_time
    def run_stencil(self, data):
        shift = np.zeros(3, dtype=int)
        shift[self.axis] = 1
        data.out[self.inner_slice()] = (data.inp[self.inner_slice(shift)] +
                                        data.inp[self.inner_slice(-shift)]) / 2


class Laplacian(LaplacianStencil):
    @timing.return_time
    def run_stencil(self, data):
        along_axes = (self.along_x, self.along_y, self.along_z)
        coeff = 2 * sum(along_axes)
        data.out[self.inner_slice()] = coeff * data.inp[self.inner_slice()]
        for axis, apply_along_axis in enumerate(along_axes):
            if apply_along_axis:
                shift = np.zeros(3, dtype=int)
                shift[axis] = 1
                data.out[self.inner_slice()] -= data.inp[self.inner_slice(
                    shift)]
                data.out[self.inner_slice()] -= data.inp[self.inner_slice(
                    -shift)]
