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
from .mixin import StencilMixin
from .. import base


class Basic(StencilMixin, base.HorizontalDiffusionStencil):
    def setup(self):
        from jax import jit, numpy as jnp

        super().setup()

        @jit
        def stencil(inp, coeff, out):
            inpi = inp[self.inner_slice(expand=(2, 2, 0))]
            coeffi = coeff[self.inner_slice(expand=(2, 2, 0))]
            lap = 4 * inpi[1:-1, 1:-1, :] - (
                inpi[2:, 1:-1, :] + inpi[:-2, 1:-1, :] + inpi[1:-1, 2:, :] +
                inpi[1:-1, :-2, :])

            flx = lap[1:, 1:-1, :] - lap[:-1, 1:-1, :]
            flx = jnp.where(
                flx * (inpi[2:-1, 2:-2, :] - inpi[1:-2, 2:-2, :]) > 0, 0, flx)

            fly = lap[1:-1, 1:, :] - lap[1:-1, :-1, :]
            fly = jnp.where(
                fly * (inpi[2:-2, 2:-1, :] - inpi[2:-2, 1:-2, :]) > 0, 0, fly)

            out = inpi[2:-2, 2:-2, :] - coeffi[2:-2, 2:-2, :] * (flx[
                1:, :, :] - flx[:-1, :, :] + fly[:, 1:, :] - fly[:, :-1, :])
            out = jnp.pad(out, self.halo, mode='empty')
            return inp, coeff, out

        self.stencil = stencil


class Vmapped(StencilMixin, base.HorizontalDiffusionStencil):
    def setup(self):
        from jax import jit, numpy as jnp, vmap

        super().setup()

        def plane(inp, coeff):
            lap = 4 * inp[1:-1, 1:-1] - (inp[2:, 1:-1] + inp[:-2, 1:-1] +
                                         inp[1:-1, 2:] + inp[1:-1, :-2])

            flx = lap[1:, 1:-1] - lap[:-1, 1:-1]
            flx = jnp.where(flx * (inp[2:-1, 2:-2] - inp[1:-2, 2:-2]) > 0, 0,
                            flx)

            fly = lap[1:-1, 1:] - lap[1:-1, :-1]
            fly = jnp.where(fly * (inp[2:-2, 2:-1] - inp[2:-2, 1:-2]) > 0, 0,
                            fly)

            return inp[2:-2, 2:-2] - coeff[2:-2, 2:-2] * (
                flx[1:, :] - flx[:-1, :] + fly[:, 1:] - fly[:, :-1])

        @jit
        def stencil(inp, coeff, out):
            inpi = inp[self.inner_slice(expand=(2, 2, 0))]
            coeffi = coeff[self.inner_slice(expand=(2, 2, 0))]
            out = vmap(plane, in_axes=2, out_axes=2)(inpi, coeffi)
            out = jnp.pad(out, self.halo, mode='empty')
            return inp, coeff, out

        self.stencil = stencil
