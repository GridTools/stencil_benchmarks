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
import ctypes
import os
import warnings

from stencil_benchmarks.benchmark import Benchmark, ExecutionError, Parameter
from stencil_benchmarks.tools import compilation, cpphelpers, template


class UnstructuredMixin(Benchmark):
    compiler = Parameter("compiler path", "g++")
    compiler_flags = Parameter("compiler flags", "")
    platform_preset = Parameter(
        "preset flags for specific hardware platform",
        "native",
        choices=["none", "native"],
    )
    print_code = Parameter("print generated code", False)
    dry_runs = Parameter("dry-runs before the measurement", 0)
    streaming_stores = Parameter(
        "enable streaming/non-temporal store instructions "
        "(only supported by some implementations)",
        False,
    )
    vector_size = Parameter("vector size", 1)
    perf_counter = Parameter(
        "perf performance counter in the form type:config, where valid values "
        "for type and config can be found in the man page of perf_event_open",
        "none",
    )

    def setup(self):
        super().setup()

        template_file = os.path.join(
            self.template_path(), "templates", self.template_file()
        )
        code = template.render(template_file, **self.template_args())
        code = cpphelpers.format_code(code, line_numbers=False)

        if self.print_code:
            print(cpphelpers.format_code(code, line_numbers=True))

        if self.compiler.endswith("icpc"):
            os.environ["KMP_INIT_AT_FORK"] = "0"

        self.compiled = compilation.GnuLibrary(code, self.compile_command())

        if self.verify and self.dry_runs:
            warnings.warn(
                "using --dry-runs together with verification might lead to "
                "false negatives for stencils with read-write fields"
            )

    def template_path(self):
        return os.path.dirname(os.path.abspath(__file__))

    def compile_command(self):
        command = [self.compiler]
        if self.platform_preset != "none":
            command += ["-std=c++11", "-Wall", "-DNDEBUG"]
            if self.compiler.endswith("icpc"):
                command += ["-qopenmp", "-ffreestanding", "-O3"]
            else:
                command += ["-fopenmp", "-Ofast"]

            if self.platform_preset == "native":
                command += ["-march=native", "-mtune=native", "-Ofast"]
        if self.compiler_flags:
            command += self.compiler_flags.split()
        return command

    @abc.abstractmethod
    def template_file(self):
        pass

    def template_args(self):
        if self.perf_counter == "none":
            perf_counter_type = perf_counter_config = None
        else:
            perf_counter_type, perf_counter_config = self.perf_counter.split(":")
        return dict(
            args=self.vertex_args + self.edge_args,
            ctype=compilation.dtype_cname(self.dtype),
            nbtype=compilation.dtype_cname(self.neighbor_table_dtype),
            vertex_strides=self.strides(self._data[0][0]) if self.vertex_args else None,
            edge_strides=self.strides(self._data[0][len(self.vertex_args)])
            if self.edge_args
            else None,
            nproma=self.nproma,
            nvertices=self.nvertices,
            nedges=self.nedges,
            nlevels=self.domain[2],
            v2e_strides=self.strides(self._v2e_table),
            e2v_strides=self.strides(self._e2v_table),
            v2e_max_neighbors=self._v2e_table.shape[1],
            e2v_max_neighbors=self._e2v_table.shape[1],
            skip_values=self.skip_values,
            alignment=self.alignment,
            streaming_stores=self.streaming_stores,
            vector_size=self.vector_size,
            dry_runs=self.dry_runs,
            perf_counter_type=perf_counter_type,
            perf_counter_config=perf_counter_config,
        )

    def data_ptr(self, data):
        return compilation.data_ptr(data)

    def run_stencil(self, data):
        time = ctypes.c_double()
        counter = ctypes.c_longlong()
        try:
            self.compiled.kernel(
                ctypes.byref(time),
                ctypes.byref(counter),
                self.data_ptr(self._v2e_table),
                self.data_ptr(self._e2v_table),
                *(self.data_ptr(array) for array in data),
            )
        except compilation.ExecutionError as error:
            raise ExecutionError() from error
        return dict(time=time.value, counter=counter.value)
