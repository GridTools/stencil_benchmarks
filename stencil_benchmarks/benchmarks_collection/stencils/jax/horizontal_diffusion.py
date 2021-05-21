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


class HdiffStencilMixin(StencilMixin):
    def setup_stencil(self):
        from jax import jit

        jited = jit(self.stencil_definition(), donate_argnums=2)
        self.stencil = lambda inp, coeff, out: (inp, coeff,
                                                jited(inp, coeff, out))


class Basic(HdiffStencilMixin, base.HorizontalDiffusionStencil):
    def stencil_definition(self):
        from jax import numpy as jnp

        class horizontal:
            def __init__(self, data):
                self.data = data

            def __getitem__(self_, slices):
                return self_.data[self.t(slices + (slice(None), ))]

        def stencil(inp, coeff, out):
            inpi = horizontal(inp[self.t(self.inner_slice(expand=(2, 2, 0)))])
            coeffi = horizontal(coeff[self.t(self.inner_slice(expand=(2, 2,
                                                                      0)))])
            lap = horizontal(4 * inpi[1:-1, 1:-1] -
                             (inpi[2:, 1:-1] + inpi[:-2, 1:-1] +
                              inpi[1:-1, 2:] + inpi[1:-1, :-2]))

            flx = horizontal(lap[1:, 1:-1] - lap[:-1, 1:-1])
            flx = horizontal(
                jnp.where(
                    flx[:, :] * (inpi[2:-1, 2:-2] - inpi[1:-2, 2:-2]) > 0, 0,
                    flx[:, :]))

            fly = horizontal(lap[1:-1, 1:] - lap[1:-1, :-1])
            fly = horizontal(
                jnp.where(
                    fly[:, :] *
                    (inpi[2:-2, 2:-1, :] - inpi[2:-2, 1:-2, :]) > 0, 0,
                    fly[:, :]))

            result = inpi[2:-2, 2:-2] - coeffi[2:-2, 2:-2] * (
                flx[1:, :] - flx[:-1, :] + fly[:, 1:] - fly[:, :-1])
            return out.at[self.t(self.inner_slice())].set(result)

        return stencil


class Vmapped(HdiffStencilMixin, base.HorizontalDiffusionStencil):
    def stencil_definition(self):
        from jax import numpy as jnp, vmap

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

        k_axis = self.layout[2]

        def stencil(inp, coeff, out):
            inpi = inp[self.t(self.inner_slice(expand=(2, 2, 0)))]
            coeffi = coeff[self.t(self.inner_slice(expand=(2, 2, 0)))]
            return out.at[self.t(self.inner_slice())].set(
                vmap(plane, in_axes=k_axis, out_axes=k_axis)(inpi, coeffi))

        return stencil
