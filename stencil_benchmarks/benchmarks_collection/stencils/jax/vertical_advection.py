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
from stencil_benchmarks.benchmark import Parameter, ParameterError
from .mixin import StencilMixin
from .. import base


class VadvStencilMixin(StencilMixin):
    def setup_stencil(self):
        from jax import jit

        if not self.u_only:
            raise ParameterError('Only option --u-only is supported')

        jited = jit(self.stencil_definition(), donate_argnums=3)
        self.stencil = (
            lambda ustage, upos, utens, utensstage, wcon, ccol, dcol, datacol:
            (ustage, upos, utens, jited(ustage, upos, utens, utensstage, wcon),
             wcon, ccol, dcol, datacol))


class Basic(VadvStencilMixin, base.VerticalAdvectionStencil):
    unroll_factor = Parameter('unroll factor', 1)

    def stencil_definition(self):
        from jax import lax, numpy as jnp, vmap

        dtr_stage = 3 / 20
        beta_v = 0
        bet_m = (1 - beta_v) / 2
        bet_p = (1 + beta_v) / 2

        def solve_column(ustage, upos, utens, utensstage, wcon, wcon_shift):
            def forward(carry, args):
                def first_level(args):
                    (wcon_k, wcon_kp1, wcon_shift_k, wcon_shift_kp1,
                     ustage_km1, ustage_k, ustage_kp1, upos_k, utens_k,
                     utensstage_k, ccol_km1, dcol_km1) = args

                    gcv = (wcon_shift_kp1 + wcon_kp1) / 4
                    cs = gcv * bet_m

                    ccol_k = gcv * bet_p
                    bcol = dtr_stage - ccol_k

                    correction_term = -cs * (ustage_kp1 - ustage_k)
                    dcol_k = (dtr_stage * upos_k + utens_k + utensstage_k +
                              correction_term)

                    divided = 1.0 / bcol
                    ccol_k = ccol_k * divided
                    dcol_k = dcol_k * divided

                    return ccol_k, dcol_k

                def other_levels(args):
                    (k, wcon_k, wcon_kp1, wcon_shift_k, wcon_shift_kp1,
                     ustage_km1, ustage_k, ustage_kp1, upos_k, utens_k,
                     utensstage_k, ccol_km1, dcol_km1) = args

                    gav = -(wcon_shift_k + wcon_k) / 4
                    gcv = (wcon_shift_kp1 + wcon_kp1) / 4

                    as_ = gav * bet_m
                    cs = gcv * bet_m

                    acol = gav * bet_p
                    ccol_k = gcv * bet_p
                    bcol = dtr_stage - acol - ccol_k

                    correction_term = -as_ * (ustage_km1 - ustage_k) - cs * (
                        ustage_kp1 - ustage_k)
                    dcol_k = (dtr_stage * upos_k + utens_k + utensstage_k +
                              correction_term)

                    divided = 1.0 / (bcol - ccol_km1 * acol)
                    ccol_k = ccol_k * divided
                    dcol_k = (dcol_k - dcol_km1 * acol) * divided

                    return ccol_k, dcol_k

                def last_level(args):
                    (wcon_k, wcon_kp1, wcon_shift_k, wcon_shift_kp1,
                     ustage_km1, ustage_k, ustage_kp1, upos_k, utens_k,
                     utensstage_k, ccol_km1, dcol_km1) = args

                    gav = -(wcon_shift_k + wcon_k) / 4

                    as_ = gav * bet_m

                    acol = gav * bet_p
                    bcol = dtr_stage - acol

                    correction_term = -as_ * (ustage_km1 - ustage_k)
                    dcol_k = (dtr_stage * upos_k + utens_k + utensstage_k +
                              correction_term)

                    divided = 1.0 / (bcol - ccol_km1 * acol)
                    dcol_k = (dcol_k - dcol_km1 * acol) * divided

                    return 0.0, dcol_k

                def first_or_last_level(args):
                    k = args[0]
                    return lax.cond(k == 0, first_level, last_level, args[1:])

                (k, wcon_k, wcon_shift_k, ustage_km1, ustage_k, ccol_km1,
                 dcol_km1) = carry
                (wcon_kp1, wcon_shift_kp1, ustage_kp1, upos_k, utens_k,
                 utensstage_k) = args
                ccol_k, dcol_k = lax.cond(
                    lax.bitwise_or(k == 0, k == self.domain[2] - 1),
                    first_or_last_level, other_levels,
                    (k, wcon_k, wcon_kp1, wcon_shift_k, wcon_shift_kp1,
                     ustage_km1, ustage_k, ustage_kp1, upos_k, utens_k,
                     utensstage_k, ccol_km1, dcol_km1))

                return (k + 1, wcon_kp1, wcon_shift_kp1, ustage_k, ustage_kp1,
                        ccol_k, dcol_k), (ccol_k, dcol_k)

            _, (ccol, dcol) = lax.scan(
                forward,
                (0, wcon[0], wcon_shift[0], ustage[0], ustage[0], 0.0, 0.0),
                (wcon[1:], wcon_shift[1:], ustage[1:], upos[:-1], utens[:-1],
                 utensstage[:-1]),
                unroll=self.unroll_factor)

            def backward(carry, args):
                def last_level(args):
                    upos_k, ccol_k, dcol_k, datacol_kp1 = args
                    datacol_k = dcol_k
                    return datacol_k, dtr_stage * (datacol_k - upos_k)

                def other_levels(args):
                    upos_k, ccol_k, dcol_k, datacol_kp1 = args
                    datacol_k = dcol_k - ccol_k * datacol_kp1
                    return datacol_k, dtr_stage * (datacol_k - upos_k)

                (last, datacol_kp1) = carry
                (upos_k, ccol_k, dcol_k) = args
                datacol_k, utensstage_k = lax.cond(
                    last, last_level, other_levels,
                    (upos_k, ccol_k, dcol_k, datacol_kp1))
                return (False, datacol_k), utensstage_k

            _, utensstage = lax.scan(backward, (True, 0.0),
                                     (upos[:-1], ccol, dcol),
                                     reverse=True,
                                     unroll=self.unroll_factor)
            return utensstage

        h = self.halo

        def stencil(ustage, upos, utens, utensstage, wcon):
            solver = vmap(vmap(solve_column))
            result = solver(ustage[h:-h, h:-h, h:-h + 1], upos[h:-h, h:-h,
                                                               h:-h + 1],
                            utens[h:-h, h:-h, h:-h + 1], utensstage[h:-h, h:-h,
                                                                    h:-h + 1],
                            wcon[h:-h, h:-h, h:-h + 1], wcon[h + 1:-h + 1,
                                                             h:-h, h:-h + 1])
            return utensstage.at[self.inner_slice()].set(result)

        return stencil
