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
import contextlib
import os

import numpy as np

from stencil_benchmarks.benchmark import Parameter, ParameterError
from stencil_benchmarks.tools import compilation

from ..openmp import mixin


class StencilMixin(mixin.StencilMixin):
    storage_block_size = Parameter('storage block width', 1)
    bit_indexing = Parameter('enable fancy bit-twiddling indexing', False)

    def setup(self):
        super().setup()

        if (self.halo % self.storage_block_size != 0
                or self.domain[0] % self.storage_block_size != 0):
            raise ParameterError(
                'halo and x-domain size have to be divisible by block size')

        if self.bit_indexing and any(s & (s - 1) != 0
                                     for s in self.blocked_strides):
            raise ParameterError(
                '--bit-indexing requires power-of-two strides')

    def template_path(self):
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def blocked_domain(self):
        assert self.domain[0] % self.storage_block_size == 0
        return (
            self.domain[0] // self.storage_block_size,
            self.domain[1],
            self.domain[2],
            self.storage_block_size,
        )

    @property
    def blocked_layout(self):
        return self.layout + (3, )

    @property
    def blocked_domain_with_halo(self):
        blocked_domain = np.array(self.blocked_domain)
        assert self.halo % self.storage_block_size == 0
        blocked_domain[0] += 2 * self.halo // self.storage_block_size
        blocked_domain[1:3] += 2 * self.halo
        return tuple(blocked_domain)

    @property
    def blocked_strides(self):
        return tuple(s * self.storage_block_size if l >= self.layout[0] else s
                     for s, l in zip(self.strides, self.layout)) + (1, )

    @contextlib.contextmanager
    def blocked(self, data):
        blocked_strides = (
            data.strides[0] * self.storage_block_size,
            data.strides[1],
            data.strides[2],
            data.strides[0],
        )
        assert data.size == np.product(self.blocked_domain_with_halo)
        blocked_view = np.lib.stride_tricks.as_strided(
            data, self.blocked_domain_with_halo, blocked_strides)

        blocked_data = self.alloc_field(self.blocked_domain_with_halo,
                                        self.blocked_layout,
                                        (self.halo, ) * 3 + (0, ))
        assert blocked_data.strides == tuple(s * blocked_data.itemsize
                                             for s in self.blocked_strides)
        blocked_data[...] = blocked_view
        yield blocked_data
        blocked_view[...] = blocked_data

    def data_ptr(self, data):
        assert self.halo % self.storage_block_size == 0
        offset = (self.halo // self.storage_block_size, self.halo, self.halo,
                  0)
        return compilation.data_ptr(data, offset)

    def run_stencil(self, data):
        with contextlib.ExitStack() as stack:
            blocked_data = [stack.enter_context(self.blocked(d)) for d in data]
            return super().run_stencil(blocked_data)

    def bit_indexing_mask(self):
        masks = []
        for axis in (0, 3):
            stride = self.blocked_strides[axis]
            layout = self.blocked_layout[axis]
            try:
                next_axis = self.blocked_layout.index(layout - 1)
                next_stride = self.blocked_strides[next_axis]
            except ValueError:
                next_stride = 2**(self.dtype_size * 8)
            masks.append((next_stride - 1) ^ (stride - 1))
        return f'0x{masks[0] | masks[1]:0x}'

    def template_args(self):
        return dict(**super().template_args(),
                    storage_block_size=self.storage_block_size,
                    blocked_domain=self.blocked_domain,
                    blocked_strides=self.blocked_strides,
                    bit_indexing=self.bit_indexing,
                    bit_indexing_mask=self.bit_indexing_mask())
