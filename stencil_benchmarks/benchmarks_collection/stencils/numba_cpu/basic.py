from .. import base
from ....tools import timing

# pylint: disable=not-an-iterable


class Copy(base.CopyStencil):
    def setup(self):
        import numba

        super().setup()

        halo = self.halo
        domain = self.domain

        @numba.njit(parallel=True)
        def kernel(inp, out):
            for k in numba.prange(halo, domain[2] + halo):
                for j in range(halo, domain[1] + halo):
                    for i in range(halo, domain[0] + halo):
                        out[i, j, k] = inp[i, j, k]

        self.kernel = kernel

    @timing.return_time
    def run_stencil(self, data):
        self.kernel(*data)


class Laplacian(base.LaplacianStencil):
    def setup(self):
        import numba

        super().setup()

        halo = self.halo
        domain = self.domain

        along = (self.along_x, self.along_y, self.along_z)
        if along == (True, True, False):

            @numba.njit(parallel=True)
            def kernel(inp, out):
                for k in numba.prange(halo, domain[2] + halo):
                    for j in range(halo, domain[1] + halo):
                        for i in range(halo, domain[0] + halo):
                            out[i, j, k] = 4 * inp[i, j, k] - (
                                inp[i - 1, j, k] + inp[i + 1, j, k] +
                                inp[i, j - 1, k] + inp[i, j + 1, k])
        elif along == (True, True, True):

            @numba.njit(parallel=True)
            def kernel(inp, out):
                for k in numba.prange(halo, domain[2] + halo):
                    for j in range(halo, domain[1] + halo):
                        for i in range(halo, domain[0] + halo):
                            out[i, j, k] = 6 * inp[i, j, k] - (
                                inp[i - 1, j, k] + inp[i + 1, j, k] +
                                inp[i, j - 1, k] + inp[i, j + 1, k] +
                                inp[i, j, k - 1] + inp[i, j, k + 1])
        else:
            raise NotImplementedError()

        self.kernel = kernel

    @timing.return_time
    def run_stencil(self, data):
        self.kernel(*data)
