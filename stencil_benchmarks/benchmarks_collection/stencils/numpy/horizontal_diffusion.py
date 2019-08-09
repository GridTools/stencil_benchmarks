import numpy as np

from ..base import HorizontalDiffusionStencil


class HorizontalDiffusion(HorizontalDiffusionStencil):
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
