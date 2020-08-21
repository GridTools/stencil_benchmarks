# Stencil Benchmarks
#
# Copyright (c) 2017-2020, ETH Zurich and MeteoSwiss
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

from ..base import HorizontalDiffusionStencil
from ....tools import timing


class OnTheFly(HorizontalDiffusionStencil):
    @timing.return_time
    def run_stencil(self, data):
        inp, coeff, out = data

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
