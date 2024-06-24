# Stencil Benchmarks
#
# Copyright (c) 2017-2021, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# SPDX-License-Identifier: BSD-3-Clause
from ....tools import timing
from .. import base

# pylint: disable=not-an-iterable


class Copy(base.CopyStencil):
    def setup(self):
        import numba

        super().setup()

        hi, hj, hk = self.halo
        domain = self.domain

        @numba.njit(parallel=True)
        def kernel(inp, out):
            for k in numba.prange(hk, domain[2] + hk):
                for j in range(hj, domain[1] + hj):
                    for i in range(hi, domain[0] + hi):
                        out[i, j, k] = inp[i, j, k]

        self.kernel = kernel

    @timing.return_time
    def run_stencil(self, data):
        self.kernel(*data)


class Laplacian(base.LaplacianStencil):
    def setup(self):
        import numba

        super().setup()

        hi, hj, hk = self.halo
        domain = self.domain

        along = (self.along_x, self.along_y, self.along_z)
        if along == (True, True, False):

            @numba.njit(parallel=True)
            def kernel(inp, out):
                for k in numba.prange(hk, domain[2] + hk):
                    for j in range(hj, domain[1] + hj):
                        for i in range(hi, domain[0] + hi):
                            out[i, j, k] = 4 * inp[i, j, k] - (
                                inp[i - 1, j, k] + inp[i + 1, j, k] +
                                inp[i, j - 1, k] + inp[i, j + 1, k])
        elif along == (True, True, True):

            @numba.njit(parallel=True)
            def kernel(inp, out):
                for k in numba.prange(hk, domain[2] + hk):
                    for j in range(hj, domain[1] + hj):
                        for i in range(hi, domain[0] + hi):
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
