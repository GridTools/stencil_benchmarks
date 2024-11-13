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
import re
import warnings
from pathlib import Path

from ...benchmark import Benchmark, ExecutionError, Parameter, ParameterError
from ...tools import compilation, cpphelpers, template


class Native(Benchmark):
    array_size = Parameter("number of elements in arrays", 10000000)
    ntimes = Parameter("number of runs", 10)
    block_size = Parameter("threads per block", 1024)
    dtype = Parameter("data type in NumPy format, e.g. float32 or float64", "float64")
    compiler = Parameter("compiler path", dtype=str, nargs=1)
    compiler_flags = Parameter("compiler flags", "")
    axis = Parameter("compute grid dimension to use", "x", choices=["x", "y", "z"])
    vector_size = Parameter("vector size", 1)
    explicit_vectorization = Parameter(
        "use float2, float3, float4 types, "
        "otherwise just add a loop and let the compiler vectorize",
        True,
    )
    unroll_factor = Parameter("loop unroll factor (in addition to vectorization)", 1)
    launch_bounds = Parameter("specify launch bounds", True)
    index_type = Parameter("index data type", "std::size_t")
    streaming_stores = Parameter("use streaming store instructions", False)
    streaming_loads = Parameter("use streaming load instructions", False)
    store_cache_modifier = Parameter(
        "PTX cache modifier for stores", "", choices=["", "wb", "cg", "cs", "wt"]
    )
    load_cache_modifier = Parameter(
        "PTX cache modifier for loads", "", choices=["", "cg", "ca", "cs", "lu", "cv"]
    )
    print_code = Parameter("print code", False)
    verify = Parameter("verify results", True)

    def setup(self):
        super().setup()

        if self.streaming_loads and self.load_cache_modifier:
            raise ParameterError(
                "streaming loads can not be combined with load cache modifier"
            )
        if self.streaming_stores and self.store_cache_modifier:
            raise ParameterError(
                "streaming stores can not be combined with store cache modifier"
            )

        elements_per_block = self.block_size * self.vector_size * self.unroll_factor
        if self.array_size % elements_per_block:
            warnings.warn("adapting array size to match block and vector sizes")
        self.array_size = (
            (self.array_size + elements_per_block - 1) // elements_per_block
        ) * elements_per_block

        template_file = Path(__file__).parent.resolve() / "cuda_hip.j2"
        code = template.render(template_file, **self.template_args())
        if self.print_code:
            print(cpphelpers.format_code(code))
        self.compiled = compilation.GnuLibrary(
            code, self.compile_command(), extension=".cu"
        )

    def compile_command(self):
        command = [self.compiler]
        if self.compiler_flags:
            command += self.compiler_flags.split()
        return command

    def template_args(self):
        return dict(
            array_size=self.array_size,
            axis=self.axis,
            block_size=self.block_size,
            ctype=compilation.dtype_cname(self.dtype),
            ntimes=self.ntimes,
            vector_size=self.vector_size,
            explicit_vectorization=self.explicit_vectorization,
            unroll_factor=self.unroll_factor,
            launch_bounds=self.launch_bounds,
            index_type=self.index_type,
            streaming_loads=self.streaming_loads,
            streaming_stores=self.streaming_stores,
            store_cache_modifier=self.store_cache_modifier,
            load_cache_modifier=self.load_cache_modifier,
            verify=self.verify,
        )

    def run(self):
        try:
            output = self.compiled.run()
        except compilation.ExecutionError as error:
            raise ExecutionError(*error.args) from error

        regex = re.compile(
            r"(Copy|Scale|Add|Triad): +"
            r"([0-9.]+) +([0-9.]+) +"
            r"([0-9.]+) +([0-9.]+)"
        )
        results = []
        for match in regex.finditer(output):
            results.append(
                {
                    "name": match.group(1).lower(),
                    "bandwidth": float(match.group(2)),
                    "avg-time": float(match.group(3)),
                    "time": float(match.group(4)),
                    "max-time": float(match.group(5)),
                }
            )

        return results
