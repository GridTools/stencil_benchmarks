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
import numpy as np

from ....tools import timing
from ..base import (
    CopyStencil,
    LaplacianStencil,
    OnesidedAverageStencil,
    SymmetricAverageStencil,
)


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
        data.out[self.inner_slice()] = (
            data.inp[self.inner_slice(shift)] + data.inp[self.inner_slice()]
        ) / 2


class SymmetricAverage(SymmetricAverageStencil):
    @timing.return_time
    def run_stencil(self, data):
        shift = np.zeros(3, dtype=int)
        shift[self.axis] = 1
        data.out[self.inner_slice()] = (
            data.inp[self.inner_slice(shift)] + data.inp[self.inner_slice(-shift)]
        ) / 2


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
                data.out[self.inner_slice()] -= data.inp[self.inner_slice(shift)]
                data.out[self.inner_slice()] -= data.inp[self.inner_slice(-shift)]
