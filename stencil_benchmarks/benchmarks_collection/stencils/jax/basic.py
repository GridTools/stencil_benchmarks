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
import numpy as np

from stencil_benchmarks.benchmark import Parameter

from .. import base
from .mixin import StencilMixin


class BasicStencilMixin(StencilMixin):
    def setup_stencil(self):
        from jax import jit

        jited = jit(self.stencil_definition(), donate_argnums=1)
        self.stencil = lambda inp, out: (inp, jited(inp, out))


class Empty(BasicStencilMixin, base.EmptyStencil):
    def stencil_definition(self):
        def stencil(inp, out):
            return out

        return stencil


class Copy(BasicStencilMixin, base.CopyStencil):
    def stencil_definition(self):
        inner = self.t(self.inner_slice())

        def stencil(inp, out):
            return out.at[inner].set(inp[inner])

        return stencil


class OnesidedAverage(BasicStencilMixin, base.OnesidedAverageStencil):
    def stencil_definition(self):
        inner = self.t(self.inner_slice())
        shift = np.zeros(3, dtype=int)
        shift[self.axis] = 1
        shifted = self.t(self.inner_slice(shift))

        def stencil(inp, out):
            return out.at[inner].set((inp[inner] + inp[shifted]) / 2)

        return stencil


class SymmetricAverage(BasicStencilMixin, base.SymmetricAverageStencil):
    def stencil_definition(self):
        inner = self.t(self.inner_slice())
        shift = np.zeros(3, dtype=int)
        shift[self.axis] = 1
        left = self.t(self.inner_slice(shift))
        right = self.t(self.inner_slice(-shift))

        def stencil(inp, out):
            return out.at[inner].set((inp[left] + inp[right]) / 2)

        return stencil


class Laplacian(BasicStencilMixin, base.LaplacianStencil):
    implementation = Parameter(
        "jax implementation to use", "set", choices=["pad", "set", "roll", "convolve"]
    )

    def stencil_definition(self):
        from jax import numpy as jnp
        from jax import scipy as jsp

        along_axes = (self.along_x, self.along_y, self.along_z)

        if self.implementation in ("pad", "set"):
            shifts = []
            for axis, apply_along_axis in enumerate(along_axes):
                if apply_along_axis:
                    shift = np.zeros(3, dtype=int)
                    shift[axis] = 1
                    left = self.t(self.inner_slice(shift))
                    center = self.t(self.inner_slice())
                    right = self.t(self.inner_slice(-shift))
                    shifts.append((left, center, right))

            def stencil(inp, out):
                result = sum(
                    2 * inp[center] - inp[left] - inp[right]
                    for left, center, right in shifts
                )
                if self.implementation == "pad":
                    return jnp.pad(result, self.halo, mode="empty")
                else:
                    return out.at[center].set(result)
        elif self.implementation == "roll":
            axes = tuple(
                i for i, along_axis in enumerate(self.t(along_axes)) if along_axis
            )

            def stencil(inp, out):
                return out.at[...].set(
                    sum(
                        2 * inp - jnp.roll(inp, 1, axis) - jnp.roll(inp, -1, axis)
                        for axis in axes
                    )
                )
        elif self.implementation == "convolve":
            window = np.zeros((3, 3, 3))
            for axis, apply_along_axis in enumerate(along_axes):
                if apply_along_axis:
                    slices = tuple(slice(None) if i == axis else 1 for i in range(3))
                    window[slices] += np.array([-1, 2, -1])
            slices = tuple(
                slice(None) if apply_along_axis else slice(1, 2)
                for apply_along_axis in along_axes
            )
            window = window.transpose(self.t((0, 1, 2)))[slices]

            def stencil(inp, out):
                return out.at[...].set(jsp.signal.convolve(inp, window, mode="same"))

        return stencil
