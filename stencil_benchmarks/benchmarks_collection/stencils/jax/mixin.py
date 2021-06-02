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
import time
import warnings

from stencil_benchmarks.benchmark import Benchmark, Parameter


class StencilMixin(Benchmark):
    dry_runs = Parameter('stencil dry-runs before the measurement', 0)
    print_jaxpr = Parameter('print resulting jaxpr', False)

    def setup(self):
        super().setup()

        if self.verify and self.dry_runs:
            warnings.warn(
                'using --dry-runs together with verification might lead to '
                'false negatives for stencils with read-write fields')

        self.setup_stencil()

        self._printed = False

    @abc.abstractmethod
    def setup_stencil(self):
        ...

    def t(self, s):
        assert isinstance(s, tuple)
        return tuple(s[self.layout.index(i)] for i in range(3))

    def s(self, lo_i=0, hi_i=0, lo_j=0, hi_j=0, lo_k=0, hi_k=0):
        def axis_slice(lo, hi, halo):
            lo += halo
            hi -= halo
            return slice(lo if lo > 0 else None, hi if hi < 0 else None)

        return self.t(
            tuple(
                axis_slice(lo, hi, h)
                for lo, hi, h in zip((lo_i, lo_j, lo_k), (hi_i, hi_j,
                                                          hi_k), self.halo)))

    def to_device(self, array):
        from jax import numpy as npy
        transposed = array.transpose(self.t((0, 1, 2)))
        return npy.array(transposed, dtype=array.dtype)

    def run_stencil(self, data):
        from jax import make_jaxpr

        device_data = [self.to_device(d) for d in data]

        if self.print_jaxpr and not self._printed:
            print(make_jaxpr(self.stencil)(*device_data).jaxpr)
            self._printed = True

        for dd in device_data:
            dd.block_until_ready()
        for _ in range(self.dry_runs):
            device_data = self.stencil(*device_data)
        for dd in device_data:
            dd.block_until_ready()

        start = time.perf_counter()
        device_data = self.stencil(*device_data)
        for dd in device_data:
            dd.block_until_ready()
        end = time.perf_counter()

        device_data = [dd.transpose(self.layout) for dd in device_data]
        for dd in device_data:
            dd.copy_to_host_async()

        for dd, d in zip(device_data, data):
            dd.block_until_ready()
            d[...] = dd
        return end - start
