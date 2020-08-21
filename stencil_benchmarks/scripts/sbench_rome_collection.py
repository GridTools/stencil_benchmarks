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

import click
import numpy as np

from stencil_benchmarks.benchmarks_collection.stencils.openmp import (
    basic, horizontal_diffusion as hdiff, vertical_advection as vadv)
from stencil_benchmarks.tools.multirun import (Configuration,
                                               run_scaling_benchmark,
                                               truncate_block_size_to_domain)
from stencil_benchmarks.tools import alloc


@click.group()
def main():
    pass


def common_kwargs(options=None, **overrides):
    kwargs = dict(platform_preset='native',
                  alignment=alloc.l1_dcache_linesize(),
                  verify=False,
                  offset_allocations=True)
    kwargs.update(overrides)
    for o in options:
        name, value = o.split('=', 1)
        name = name.replace('-', '_')
        try:
            value = bool(value)
        except ValueError:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        kwargs[name] = value
    return kwargs


@main.command()
@click.argument('output', type=click.Path())
@click.option('--executions', '-e', type=int, default=101)
@click.option('--dtype', '-d', default='float32')
@click.option('--option', '-o', multiple=True)
def basic_bandwidth(output, executions, dtype, option):
    vector_size = 32 // np.dtype(dtype).itemsize
    kwargs = common_kwargs(option,
                           dtype=dtype,
                           vector_size=vector_size,
                           loop='3D-blocked-vec',
                           halo=1,
                           block_size=(1024, 16, 1),
                           streaming_stores=True)

    stream_kwargs = kwargs.copy()
    stream_kwargs.update(loop='1D-vec', halo=0)

    configurations = [
        Configuration(basic.Copy, name='stream', **stream_kwargs),
        Configuration(basic.Copy, name='copy', **kwargs),
        Configuration(basic.OnesidedAverage, name='avg-i', axis=0, **kwargs),
        Configuration(basic.OnesidedAverage, name='avg-j', axis=1, **kwargs),
        Configuration(basic.OnesidedAverage, name='avg-k', axis=2, **kwargs),
        Configuration(basic.SymmetricAverage,
                      name='sym-avg-i',
                      axis=0,
                      **kwargs),
        Configuration(basic.SymmetricAverage,
                      name='sym-avg-j',
                      axis=1,
                      **kwargs),
        Configuration(basic.SymmetricAverage,
                      name='sym-avg-k',
                      axis=2,
                      **kwargs),
        Configuration(basic.Laplacian,
                      name='lap-ij',
                      along_x=True,
                      along_y=True,
                      along_z=False,
                      **kwargs)
    ]
    table = run_scaling_benchmark(
        configurations,
        executions,
        preprocess_args=truncate_block_size_to_domain)
    table.to_csv(output)


@main.command()
@click.argument('output', type=click.Path())
@click.option('--executions', '-e', type=int, default=101)
@click.option('--dtype', '-d', default='float32')
@click.option('--option', '-o', multiple=True)
def horizontal_diffusion_bandwidth(output, executions, dtype, option):
    vector_size = 32 // np.dtype(dtype).itemsize
    kwargs = common_kwargs(option,
                           dtype=dtype,
                           vector_size=vector_size,
                           streaming_stores=True,
                           block_size=(1024, 16, 1))

    configurations = [
        Configuration(hdiff.ClassicVec, **kwargs),
        Configuration(hdiff.OnTheFlyVec, **kwargs),
        Configuration(hdiff.MinimumMem, **kwargs)
    ]

    table = run_scaling_benchmark(
        configurations,
        executions,
        preprocess_args=truncate_block_size_to_domain)
    table.to_csv(output)


@main.command()
@click.argument('output', type=click.Path())
@click.option('--executions', '-e', type=int, default=101)
@click.option('--dtype', '-d', default='float32')
@click.option('--option', '-o', multiple=True)
def vertical_advection_bandwidth(output, executions, dtype, option):
    vector_size = 32 // np.dtype(dtype).itemsize
    kwargs = common_kwargs(option, dtype=dtype, vector_size=vector_size)

    configurations = [
        Configuration(vadv.KMiddleVec,
                      **kwargs,
                      block_size=(128, 1),
                      streaming_stores=True),
        Configuration(vadv.KInnermostVec, **kwargs, block_size=(64, 1)),
        Configuration(vadv.KInnermostBlockVec,
                      **kwargs,
                      block_size=(16, 1),
                      prefetch_distance=4,
                      streaming_stores=True)
    ]

    table = run_scaling_benchmark(
        configurations,
        executions,
        preprocess_args=truncate_block_size_to_domain)
    table.to_csv(output)


if __name__ == '__main__':
    main()
