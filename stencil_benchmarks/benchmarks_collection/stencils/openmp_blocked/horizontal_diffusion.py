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
from stencil_benchmarks.benchmarks_collection.stencils import base

from .basic import BasicStencilMixin


class OnTheFly(BasicStencilMixin, base.HorizontalDiffusionStencil):
    def stencil_body(self):
        return f'''const auto inp_ij = inp[{self.index(0, 0, 0)}];
                const auto inp_im1j = inp[{self.index(-1, 0, 0)}];
                const auto inp_ip1j = inp[{self.index(1, 0, 0)}];
                const auto inp_ijm1 = inp[{self.index(0, -1, 0)}];
                const auto inp_ijp1 = inp[{self.index(0, 1, 0)}];
                const auto inp_im2j = inp[{self.index(-2, 0, 0)}];
                const auto inp_im1jm1 = inp[{self.index(-1, -1, 0)}];
                const auto inp_im1jp1 = inp[{self.index(-1, 1, 0)}];
                const auto inp_ip2j = inp[{self.index(2, 0, 0)}];
                const auto inp_ip1jm1 = inp[{self.index(1, -1, 0)}];
                const auto inp_ip1jp1 = inp[{self.index(1, 1, 0)}];
                const auto inp_ijm2 = inp[{self.index(0, -2, 0)}];
                const auto inp_ijp2 = inp[{self.index(0, 2, 0)}];

                const auto lap_ij = 4 * inp_ij - inp_im1j
                                               - inp_ip1j
                                               - inp_ijm1
                                               - inp_ijp1;
                const auto lap_imj = 4 * inp_im1j - inp_im2j
                                                  - inp_ij
                                                  - inp_im1jm1
                                                  - inp_im1jp1;
                const auto lap_ipj = 4 * inp_ip1j - inp_ij
                                                  - inp_ip2j
                                                  - inp_ip1jm1
                                                  - inp_ip1jp1;
                const auto lap_ijm = 4 * inp_ijm1 - inp_im1jm1
                                                  - inp_ip1jm1
                                                  - inp_ijm2
                                                  - inp_ij;
                const auto lap_ijp = 4 * inp_ijp1 - inp_im1jp1
                                                  - inp_ip1jp1
                                                  - inp_ij
                                                  - inp_ijp2;

                auto flx_ij = lap_ipj - lap_ij;
                flx_ij = flx_ij * (inp_ip1j - inp_ij) > 0 ? 0 : flx_ij;

                auto flx_imj = lap_ij - lap_imj;
                flx_imj = flx_imj * (inp_ij - inp_im1j) > 0 ? 0 : flx_imj;

                auto fly_ij = lap_ijp - lap_ij;
                fly_ij = fly_ij * (inp_ijp1 - inp_ij) > 0 ? 0 : fly_ij;

                auto fly_ijm = lap_ij - lap_ijm;
                fly_ijm = fly_ijm * (inp_ij - inp_ijm1) > 0 ? 0 : fly_ijm;

                out[{self.index(0, 0, 0)}] = inp_ij
                                           - coeff[{self.index(0, 0, 0)}] * (
                                                      flx_ij - flx_imj +
                                                      fly_ij - fly_ijm);
                '''
