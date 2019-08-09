import numba

from .. import base

# pylint: disable=not-an-iterable


@numba.njit(parallel=True)
def copy_apply_kernel(domain, halo, inp, out):
    domain_x, domain_y, domain_z = domain
    for i in numba.prange(halo, domain_x + halo):
        for j in range(halo, domain_y + halo):
            for k in range(halo, domain_z + halo):
                out[i, j, k] = inp[i, j, k]


class Copy(base.CopyStencil):
    def run_stencil(self, data_set):
        inp, out = self.inouts[data_set]
        copy_apply_kernel(self.domain, self.halo, inp, out)


@numba.njit(parallel=True)
def laplacian_ij_apply_kernel(domain, halo, inp, out):
    domain_x, domain_y, domain_z = domain
    for i in numba.prange(halo, domain_x + halo):
        for j in range(halo, domain_y + halo):
            for k in range(halo, domain_z + halo):
                out[i, j, k] = 4 * inp[i, j, k] - (
                    inp[i - 1, j, k] + inp[i + 1, j, k] + inp[i, j - 1, k] +
                    inp[i, j + 1, k])


@numba.njit(parallel=True)
def laplacian_ijk_apply_kernel(domain, halo, inp, out):
    domain_x, domain_y, domain_z = domain
    for i in numba.prange(halo, domain_x + halo):
        for j in range(halo, domain_y + halo):
            for k in range(halo, domain_z + halo):
                out[i, j, k] = 6 * inp[i, j, k] - (
                    inp[i - 1, j, k] + inp[i + 1, j, k] + inp[i, j - 1, k] +
                    inp[i, j + 1, k] + inp[i, j, k - 1] + inp[i, j, k + 1])


class Laplacian(base.LaplacianStencil):
    def run_stencil(self, data_set):
        inp, out = self.inouts[data_set]
        along = (self.along_x, self.along_y, self.along_z)
        if along == (True, True, False):
            laplacian_ij_apply_kernel(self.domain, self.halo, inp, out)
        elif along == (True, True, True):
            laplacian_ijk_apply_kernel(self.domain, self.halo, inp, out)
        else:
            raise NotImplementedError()
