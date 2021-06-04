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
        n = self.storage_block_size // self.vector_size
        return (f'for (index_t v = 0; v < {n}; ++v)\n'
                f'    storent(&out[idx + v * {self.vector_size}], \n'
                f'            load(&inp[idx + v * {self.vector_size}]));')


class OnesidedAverage(BasicStencilMixin, base.OnesidedAverageStencil):
    def stencil_body(self):
        n = self.storage_block_size // self.vector_size
        o = f'v * {self.vector_size}'
        stride = self.blocked_strides[self.axis]
        if self.axis == 0:
            p1 = ', '.join(f'{i + 1}' for i in range(self.vector_size))
            return (f'vec_t inp_c, inp_pv;\n'
                    f'for (index_t v = 0; v < {n}; ++v) {{\n'
                    f'    if (v == 0) inp_c = load(&inp[idx]);\n'
                    f'    else inp_c = inp_pv;\n'
                    f'    if (v == {n - 1})\n'
                    f'        inp_pv = load(&inp[idx + {stride}]);\n'
                    f'    else\n'
                    f'        inp_pv = load(\n'
                    f'            &inp[idx + (v + 1) * {self.vector_size}]);\n'
                    f'    vec_t inp_p1 = shuffle(inp_c, inp_pv, {p1});\n'
                    f'    storent(&out[idx + {o}], (inp_c + inp_p1) / 2);\n'
                    f'}}\n')
        else:
            return (f'for (index_t v = 0; v < {n}; ++v) {{\n'
                    f'    vec_t inp_c = load(&inp[idx + {o}]);\n'
                    f'    vec_t inp_p1 = load(&inp[idx + {stride} + {o}]);\n'
                    f'    storent(&out[idx + {o}], (inp_c + inp_p1) / 2);\n'
                    f'}}\n')


class SymmetricAverage(BasicStencilMixin, base.SymmetricAverageStencil):
    def stencil_body(self):
        n = self.storage_block_size // self.vector_size
        o = f'v * {self.vector_size}'
        stride = self.blocked_strides[self.axis]
        if self.axis == 0:
            m1 = ', '.join(f'{i - 1}' for i in range(self.vector_size, 2 *
                                                     self.vector_size))
            p1 = ', '.join(f'{i + 1}' for i in range(self.vector_size))
            return (f'vec_t inp_mv, inp_c, inp_pv;\n'
                    f'for (index_t v = 0; v < {n}; ++v) {{\n'
                    f'    if (v == 0) {{\n'
                    f'        inp_mv = load(&inp[idx - {stride}\n'
                    f'            + {n - 1} * {self.vector_size}]);\n'
                    f'        inp_c = load(&inp[idx]);\n'
                    f'    }} else {{\n'
                    f'        inp_mv = inp_c;\n'
                    f'        inp_c = inp_pv;\n'
                    f'    }}\n'
                    f'    if (v == {n - 1})\n'
                    f'        inp_pv = load(&inp[idx + {stride}]);\n'
                    f'    else\n'
                    f'        inp_pv = load(\n'
                    f'            &inp[idx + (v + 1) * {self.vector_size}]);\n'
                    f'    vec_t inp_m1 = shuffle(inp_mv, inp_c, {m1});\n'
                    f'    vec_t inp_p1 = shuffle(inp_c, inp_pv, {p1});\n'
                    f'    storent(&out[idx + {o}], (inp_m1 + inp_p1) / 2);\n'
                    f'}}\n')
        else:
            return (f'for (index_t v = 0; v < {n}; ++v) {{\n'
                    f'    vec_t inp_m1 = load(&inp[idx - {stride} + {o}]);\n'
                    f'    vec_t inp_p1 = load(&inp[idx + {stride} + {o}]);\n'
                    f'    storent(&out[idx + {o}], (inp_m1 + inp_p1) / 2);\n'
                    f'}}\n')


class Laplacian(BasicStencilMixin, base.LaplacianStencil):
    def stencil_body(self):
        n = self.storage_block_size // self.vector_size
        o = f'v * {self.vector_size}'
        code = ''
        if self.along_x:
            code += 'vec_t inp_imvjk, inp_ijk, inp_ipvjk;\n'
        code += f'for (index_t v = 0; v < {n}; ++v) {{\n'
        terms = []
        if self.along_x:
            stride = self.blocked_strides[0]
            m1 = ', '.join(f'{i - 1}' for i in range(self.vector_size, 2 *
                                                     self.vector_size))
            p1 = ', '.join(f'{i + 1}' for i in range(self.vector_size))
            code += (
                f'    if (v == 0) {{\n'
                f'        inp_imvjk = load(&inp[idx - {stride}\n'
                f'            + {n - 1} * {self.vector_size}]);\n'
                f'        inp_ijk = load(&inp[idx]);\n'
                f'    }} else {{\n'
                f'        inp_imvjk = inp_ijk;\n'
                f'        inp_ijk = inp_ipvjk;\n'
                f'    }}\n'
                f'    if (v == {n - 1})\n'
                f'        inp_ipvjk = load(&inp[idx + {stride}]);\n'
                f'    else\n'
                f'        inp_ipvjk = load(\n'
                f'            &inp[idx + (v + 1) * {self.vector_size}]);\n'
                f'    vec_t inp_im1jk = shuffle(inp_imvjk, inp_ijk, {m1});\n'
                f'    vec_t inp_ip1jk = shuffle(inp_ijk, inp_ipvjk, {p1});\n')
            terms += ['inp_im1jk', 'inp_ip1jk']
        else:
            code += f'vec_t inp_ijk = load(&inp[idx + {o}]);\n'
        if self.along_y:
            stride = self.blocked_strides[1]
            code += (
                f'    vec_t inp_ijm1k = load(&inp[idx - {stride} + {o}]);\n'
                f'    vec_t inp_ijp1k = load(&inp[idx + {stride} + {o}]);\n')
            terms += ['inp_ijm1k', 'inp_ijp1k']
        if self.along_z:
            stride = self.blocked_strides[2]
            code += (
                f'    vec_t inp_ijkm1 = load(&inp[idx - {stride} + {o}]);\n'
                f'    vec_t inp_ijkp1 = load(&inp[idx + {stride} + {o}]);\n')
            terms += ['inp_ijkm1', 'inp_ijkp1']
        code += (f'    storent(&out[idx + {o}], {len(terms)} * inp_ijk - (' +
                 ' + '.join(terms) + '));'
                 f'}}\n')
        return code
