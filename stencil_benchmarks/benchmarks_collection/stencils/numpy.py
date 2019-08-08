import numpy as np

from . import base

# pylint: disable=no-member,invalid-name


class Copy1D(base.CopyStencil):
    def run_stencil(self, data_set):
        inp, out = self.inouts[data_set]
        out[:] = inp[:]


class Copy(base.CopyStencil):
    def run_stencil(self, data_set):
        inp, out = self.inouts[data_set]
        out[self.inner_slice()] = inp[self.inner_slice()]


class OnesidedAverage(base.OnesidedAverageStencil):
    def run_stencil(self, data_set):
        inp, out = self.inouts[data_set]
        shift = np.zeros(3, dtype=int)
        shift[self.axis] = 1
        out[self.inner_slice()] = (inp[self.inner_slice(shift)] +
                                   inp[self.inner_slice()]) / 2


class SymmetricAverage(base.SymmetricAverageStencil):
    def run_stencil(self, data_set):
        inp, out = self.inouts[data_set]
        shift = np.zeros(3, dtype=int)
        shift[self.axis] = 1
        out[self.inner_slice()] = (inp[self.inner_slice(shift)] +
                                   inp[self.inner_slice(-shift)]) / 2


class Laplacian(base.LaplacianStencil):
    def run_stencil(self, data_set):
        inp, out = self.inouts[data_set]
        along_axes = (self.along_x, self.along_y, self.along_z)
        coeff = 2 * sum(along_axes)
        out[self.inner_slice()] = coeff * inp[self.inner_slice()]
        for axis, apply_along_axis in enumerate(along_axes):
            if apply_along_axis:
                shift = np.zeros(3, dtype=int)
                shift[axis] = 1
                out[self.inner_slice()] -= inp[self.inner_slice(shift)]
                out[self.inner_slice()] -= inp[self.inner_slice(-shift)]


class HorizontalDiffusion(base.HorizontalDiffusionStencil):
    def run_stencil(self, data_set):
        inp, coeff, out = self.inouts[data_set]

        lap = 4 * inp[1:-1, 1:-1, :] - (inp[2:, 1:-1, :] + inp[:-2, 1:-1, :] +
                                        inp[1:-1, 2:, :] + inp[1:-1, :-2, :])

        flx = lap[1:, :-1, :] - lap[:-1, :-1, :]
        flx = np.where(flx * (inp[2:-1, 1:-2, :] - inp[1:-2, 1:-2, :]) > 0, 0,
                       flx)

        fly = lap[:-1, 1:, :] - lap[:-1, :-1, :]
        fly = np.where(fly * (inp[1:-2, 2:-1, :] - inp[1:-2, 1:-2, :]) > 0, 0,
                       fly)

        out[2:-2, 2:-2, :] = inp[2:-2, 2:-2, :] - coeff[2:-2, 2:-2, :] * (
            flx[1:, 1:, :] - flx[:-1, 1:, :] + fly[1:, 1:, :] -
            fly[1:, :-1, :])
