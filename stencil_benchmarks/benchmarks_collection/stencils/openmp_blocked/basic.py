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
import abc

from stencil_benchmarks.benchmarks_collection.stencils import base

from .mixin import StencilMixin


class BasicStencilMixin(StencilMixin):
    @abc.abstractmethod
    def stencil_body(self):
        pass

    def template_file(self):
        return 'basic.j2'

    def template_args(self):
        return dict(**super().template_args(), body=self.stencil_body())


class Copy(BasicStencilMixin, base.CopyStencil):
    def stencil_body(self):
        return 'VL((out[idx] = inp[idx]))'


class OnesidedAverage(BasicStencilMixin, base.OnesidedAverageStencil):
    def stencil_body(self):
        center_load = 'VD(inp_c)\nVL((inp_c[bi] = inp[idx]))\n'
        stride = self.blocked_strides[self.axis]
        if self.axis == 0:
            b = self.blocked_domain[3] - 1
            offset_load = (
                f'VD(inp_pb)\n'
                f'VL((inp_pb[bi] = inp[idx + {stride}]))\n'
                f'VD(inp_p1)\n'
                f'VL((inp_p1[bi] = bi == {b} ? inp_pb[0] : inp_c[bi + 1]))\n')
        else:
            offset_load = (f'VD(inp_p1)\n'
                           f'VL((inp_p1[bi] = inp[idx + {stride}]))\n')
        write = 'VL((out[idx] = (inp_c[bi] + inp_p1[bi]) / 2))'
        return center_load + offset_load + write


class SymmetricAverage(BasicStencilMixin, base.SymmetricAverageStencil):
    def stencil_body(self):
        stride = self.blocked_strides[self.axis]
        if self.axis == 0:
            b = self.blocked_domain[3] - 1
            loads = (
                f'VD(inp_c)\n'
                f'VL((inp_c[bi] = inp[idx]))\n'
                f'VD(inp_mb)\n'
                f'VL((inp_mb[bi] = inp[idx - {stride}]))\n'
                f'VD(inp_m1)\n'
                f'VL((inp_m1[bi] = bi == 0 ? inp_mb[{b}] : inp_c[bi - 1]))\n'
                f'VD(inp_pb)\n'
                f'VL((inp_pb[bi] = inp[idx + {stride}]))\n'
                f'VD(inp_p1)\n'
                f'VL((inp_p1[bi] = bi == {b} ? inp_pb[0] : inp_c[bi + 1]))\n')
        else:
            loads = (f'VD(inp_m1)\n'
                     f'VL((inp_m1[bi] = inp[idx - {stride}]))\n'
                     f'VD(inp_p1)\n'
                     f'VL((inp_p1[bi] = inp[idx + {stride}]))\n')
        write = 'VL((out[idx] = (inp_m1[bi] + inp_p1[bi]) / 2))'
        return loads + write


class Laplacian(BasicStencilMixin, base.LaplacianStencil):
    def stencil_body(self):
        along_axes = (self.along_x, self.along_y, self.along_z)
        coeff = 2 * sum(along_axes)
        loads = ''
        terms = []
        for axis, (name, apply_along_axis) in enumerate(zip('ijk',
                                                            along_axes)):
            if apply_along_axis:
                stride = self.blocked_strides[axis]
                if axis == 0:
                    b = self.blocked_domain[3] - 1
                    axis_loads = (
                        f'VD(inp_{name}mb)\n'
                        f'VL((inp_{name}mb[bi] = inp[idx - {stride}]))\n'
                        f'VD(inp_{name}m1)\n'
                        f'VL((inp_{name}m1[bi] = bi == 0 ? '
                        f'inp_{name}mb[{b}] : inp_c[bi - 1]))\n'
                        f'VD(inp_{name}pb)\n'
                        f'VL((inp_{name}pb[bi] = inp[idx + {stride}]))\n'
                        f'VD(inp_{name}p1)\n'
                        f'VL((inp_{name}p1[bi] = bi == {b} ? '
                        f'inp_{name}pb[0] : inp_c[bi + 1]))\n')
                else:
                    axis_loads = (f'VD(inp_{name}m1)\n'
                                  f'VL((inp_{name}m1[bi] = '
                                  f'inp[idx - {stride}]))\n'
                                  f'VD(inp_{name}p1)\n'
                                  f'VL((inp_{name}p1[bi] = '
                                  f'inp[idx + {stride}]))\n')
                loads += axis_loads
                terms += [f'inp_{name}m1[bi]', f'inp_{name}p1[bi]']

        center_load = 'VD(inp_c)\nVL((inp_c[bi] = inp[idx]))\n'
        write = (f'VL((out[idx] = {coeff} * inp_c[bi] - (' +
                 ' + '.join(terms) + ')))')
        return center_load + loads + write
