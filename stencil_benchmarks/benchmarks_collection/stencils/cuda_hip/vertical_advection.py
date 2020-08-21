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
from stencil_benchmarks.benchmark import Parameter
from stencil_benchmarks.benchmarks_collection.stencils import base

from .mixin import StencilMixin


class VerticalAdvectionMixin(StencilMixin):
    block_size = Parameter('block size', (32, 1))
    unroll_factor = Parameter(
        'force unrolling of vertical loop '
        '(-1: no enforced unrolling, 0: full unrolling,'
        ' > 0: unrolling factor)', -1)

    def template_file(self):
        return 'vertical_advection_' + type(self).__name__.lower() + '.j2'

    def template_args(self):
        return dict(**super().template_args(),
                    block_size=self.block_size,
                    unroll_factor=self.unroll_factor,
                    u_only=self.u_only)


class Classic(VerticalAdvectionMixin, base.VerticalAdvectionStencil):
    pass


class SharedMem(VerticalAdvectionMixin, base.VerticalAdvectionStencil):
    pass


class LocalMem(VerticalAdvectionMixin, base.VerticalAdvectionStencil):
    pass


class LocalMemMerged(VerticalAdvectionMixin, base.VerticalAdvectionStencil):
    pass
