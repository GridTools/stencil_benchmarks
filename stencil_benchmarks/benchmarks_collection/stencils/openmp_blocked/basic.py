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

    def index(self, off_i, off_j, off_k):
        return f'index<{off_i}, {off_j}, {off_k}>(i, j, k, bi)'


class Copy(BasicStencilMixin, base.CopyStencil):
    def stencil_body(self):
        center = self.index(0, 0, 0)
        return f'out[{center}] = inp[{center}];'


class OnesidedAverage(BasicStencilMixin, base.OnesidedAverageStencil):
    def stencil_body(self):
        center = self.index(0, 0, 0)
        right = self.index(*(1 if i == self.axis else 0 for i in range(3)))
        return f'out[{center}] = (inp[{center}] + inp[{right}]) / 2;'


class SymmetricAverage(BasicStencilMixin, base.SymmetricAverageStencil):
    def stencil_body(self):
        center = self.index(0, 0, 0)
        left = self.index(*(-1 if i == self.axis else 0 for i in range(3)))
        right = self.index(*(1 if i == self.axis else 0 for i in range(3)))
        return f'out[{center}] = (inp[{left}] + inp[{right}]) / 2;'


class Laplacian(BasicStencilMixin, base.LaplacianStencil):
    def stencil_body(self):
        along_axes = (self.along_x, self.along_y, self.along_z)
        coeff = 2 * sum(along_axes)
        code = []
        center = self.index(0, 0, 0)
        for axis, apply_along_axis in enumerate(along_axes):
            if apply_along_axis:
                left = self.index(*(-1 if i == axis else 0 for i in range(3)))
                right = self.index(*(1 if i == axis else 0 for i in range(3)))
                code.append(f'inp[{left}] + inp[{right}]')

        return f'out[{center}] = {coeff} * inp[{center}] - (' + ' + '.join(
            code) + ');'
