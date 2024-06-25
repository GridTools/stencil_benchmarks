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
import collections
import copy

import numpy as np

from ...benchmark import Benchmark, Parameter, ParameterError
from ...tools import alloc, array, parallel, validation


class Unstructured(Benchmark):
    domain = Parameter("domain size", (128, 128, 80))
    skip_values = Parameter("use neighbor table with skip/empty neighbor values", True)
    data_sets = Parameter(
        "number of data sets, if bigger than one, data sets are cycled before "
        "each execution to start with cold cache",
        1,
    )
    dtype = Parameter("data type in NumPy format, e.g. float32 or float64", "float64")
    layout = Parameter(
        "data layout, 2 means innermost dimension, 0 outermost", (2, 1, 0)
    )
    neighbor_table_layout = Parameter(
        "data layout for neighbor table, 1 means inner dimension, 0 outer", (1, 0)
    )
    neighbor_table_dtype = Parameter(
        "neighbor table dtype in NumPy format, e.g., int32 or int64", "int64"
    )
    nproma = Parameter("data/loop block size", 1)
    alignment = Parameter("data alignment in bytes", 0)
    huge_pages = Parameter(
        "use huge pages", "none", choices=["none", "transparent", "explicit"]
    )
    offset_allocations = Parameter(
        "offset allocated data by some bytes to minimize cache conflicts", False
    )
    verify = Parameter("enable verification", True)

    def setup(self):
        super().setup()
        if tuple(sorted(self.layout)) != (0, 1, 2):
            raise ParameterError(f"invalid layout specification {self.layout}")
        if self.alignment < 0:
            raise ParameterError(f"negative alignment given ({self.alignment} bytes)")
        if self.alignment % self.dtype_size != 0:
            raise ParameterError(
                f"alignment ({self.alignment} bytes) not divisible "
                f"by dtype size ({self.dtype_size} bytes)"
            )

        self._v2e_table = self.v2e_table()
        self._e2v_table = self.e2v_table()

        stencil_data = collections.namedtuple(
            "StencilData", self.vertex_args + self.edge_args
        )
        self._data = [
            stencil_data(
                *(self.random_vertex_field() for _ in self.vertex_args),
                *(self.random_edge_field() for _ in self.edge_args),
            )
            for _ in range(self.data_sets)
        ]
        self._run = 0

    def _allocate(self, nbytes):
        if self.huge_pages == "none":
            return alloc.alloc_smallpages(nbytes)
        return alloc.alloc_hugepages(nbytes, self.huge_pages == "transparent")

    def alloc_field(self, shape, layout, dtype):
        return array.alloc_array(
            shape,
            dtype,
            layout,
            self.alignment,
            alloc=self._allocate,
            apply_offset=self.offset_allocations,
        )

    def empty_vertex_field(self):
        return self.alloc_field(self.vertex_data_shape, self.layout, self.dtype)

    def empty_edge_field(self):
        return self.alloc_field(self.edge_data_shape, self.layout, self.dtype)

    def random_vertex_field(self):
        data = self.empty_vertex_field()
        parallel.random_fill(data)
        return data

    def random_edge_field(self):
        data = self.empty_edge_field()
        parallel.random_fill(data)
        return data

    @property
    def dtype_size(self):
        return np.dtype(self.dtype).itemsize

    @property
    def neighbor_table_dtype_size(self):
        return np.dtype(self.neighbor_table_dtype).itemsize

    def v2e_table(self):
        nx, ny, _ = self.domain
        shape = nx * ny, 6
        data = self.alloc_field(
            shape, self.neighbor_table_layout, self.neighbor_table_dtype
        )

        i = np.arange(nx * ny) % nx
        j = np.arange(nx * ny) // nx

        if self.skip_values:
            nxedges = (nx - 1) * ny
            nyedges = nx * (ny - 1)
            data[:, 0] = np.where(i > 0, (i - 1) + (nx - 1) * j, -1)
            data[:, 1] = np.where(i < nx - 1, i + (nx - 1) * j, -1)
            data[:, 2] = np.where(j > 0, nxedges + i + nx * (j - 1), -1)
            data[:, 3] = np.where(j < ny - 1, nxedges + i + nx * j, -1)
            data[:, 4] = np.where(
                (i > 0) & (j > 0), nxedges + nyedges + i - 1 + (nx - 1) * (j - 1), -1
            )
            data[:, 5] = np.where(
                (i < nx - 1) & (j < ny - 1), nxedges + nyedges + i + (nx - 1) * j, -1
            )
        else:
            nxedges = nyedges = nx * ny
            data[:, 0] = (i - 1) % nx + nx * j
            data[:, 1] = i + nx * j
            data[:, 2] = nxedges + i + nx * ((j - 1) % ny)
            data[:, 3] = nxedges + i + nx * j
            data[:, 4] = nxedges + nyedges + (i - 1) % nx + nx * ((j - 1) % ny)
            data[:, 5] = nxedges + nyedges + i + nx * j

        return data

    def remove_nproma(self, data):
        return np.concatenate([data[:, i, :] for i in range(data.shape[1])], axis=0)

    def e2v_table(self):
        nx, ny, _ = self.domain
        if self.skip_values:
            nxedges = (nx - 1) * ny
            nyedges = nx * (ny - 1)
            ndedges = (nx - 1) * (ny - 1)
        else:
            nxedges = nyedges = ndedges = nx * ny
        shape = nxedges + nyedges + ndedges, 2
        data = self.alloc_field(
            shape, self.neighbor_table_layout, self.neighbor_table_dtype
        )

        nex = (nx - 1, nx, nx - 1) if self.skip_values else (nx, nx, nx)
        i = np.concatenate(
            [np.arange(ntot) % n for ntot, n in zip((nxedges, nyedges, ndedges), nex)]
        )
        j = np.concatenate(
            [np.arange(ntot) // n for ntot, n in zip((nxedges, nyedges, ndedges), nex)]
        )

        s = slice(None, nxedges)
        data[s, 0] = i[s] + nx * j[s]
        data[s, 1] = i[s] + 1 + nx * j[s]
        s = slice(nxedges, nxedges + nyedges)
        data[s, 0] = i[s] + nx * j[s]
        data[s, 1] = i[s] + nx * (j[s] + 1)
        s = slice(nxedges + nyedges, None)
        data[s, 0] = i[s] + nx * j[s]
        data[s, 1] = i[s] + 1 + nx * (j[s] + 1)

        return data

    @property
    def nvertices(self):
        return self._v2e_table.shape[0]

    @property
    def nedges(self):
        return self._e2v_table.shape[0]

    @property
    def vertex_data_shape(self):
        return (
            self.nproma,
            (self.nvertices + self.nproma - 1) // self.nproma,
            self.domain[2],
        )

    @property
    def edge_data_shape(self):
        return (
            self.nproma,
            (self.nedges + self.nproma - 1) // self.nproma,
            self.domain[2],
        )

    def strides(self, data):
        return tuple(s // data.dtype.itemsize for s in data.strides)

    @property
    def data_size(self):
        return (
            (len(self.vertex_args) * self.nvertices + len(self.edge_args) * self.nedges)
            * self.domain[2]
            * self.dtype_size
        )

    @abc.abstractmethod
    def run_stencil(self, data):
        pass

    @abc.abstractmethod
    def verify_stencil(self, data_before, data_after):
        pass

    @abc.abstractproperty
    def vertex_args(self):
        pass

    @abc.abstractproperty
    def edge_args(self):
        pass

    def run(self):
        data_index = self._run % self.data_sets

        if self.verify:
            data_before = copy.deepcopy(self._data[data_index])

        result = self.run_stencil(self._data[data_index])

        if self.verify:
            self.verify_stencil(data_before, self._data[data_index])

        self._run += 1
        assert "time" in result and result["time"] > 0
        assert "bandwidth" not in result
        result["bandwidth"] = self.data_size / result["time"] / 1e9
        return result


class UnstructuredCopy(Unstructured):
    @property
    def vertex_args(self):
        return "inp", "out"

    @property
    def edge_args(self):
        return ()

    def verify_stencil(self, data_before, data_after):
        validation.check_equality("inp", data_before.inp, data_after.inp)
        validation.check_equality("out", data_after.out, data_before.inp)


class EdgeSum(Unstructured):
    @property
    def vertex_args(self):
        return ("out",)

    @property
    def edge_args(self):
        return ("inp",)

    def verify_stencil(self, data_before, data_after):
        validation.check_equality("inp", data_before.inp, data_after.inp)
        out = self.remove_nproma(data_after.out)[: self.nvertices, :]
        inp = self.remove_nproma(data_before.inp)[: self.nedges, :]
        validation.check_equality(
            "out",
            out,
            np.sum(
                np.where(
                    (self._v2e_table != -1)[:, :, np.newaxis],
                    inp[self._v2e_table, :],
                    0,
                ),
                axis=1,
            ),
        )
