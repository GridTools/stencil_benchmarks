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

from stencil_benchmarks.benchmarks_collection.stencils.cuda_hip import (
    basic, horizontal_diffusion as hdiff, vertical_advection as vadv)
from stencil_benchmarks.tools.multirun import (Configuration,
                                               run_scaling_benchmark,
                                               truncate_block_size_to_domain,
                                               default_kwargs)


@click.group()
def main():
    pass


common_kwargs = default_kwargs(backend='cuda',
                               compiler='nvcc',
                               gpu_architecture='sm_70',
                               verify=False,
                               dry_runs=1,
                               gpu_timers=True,
                               alignment=128,
                               dtype='float32')


@main.command()
@click.argument('output', type=click.Path())
@click.option('--executions', '-e', type=int, default=101)
@click.option('--option', '-o', multiple=True)
def basic_bandwidth(output, executions, option):
    kwargs = common_kwargs(
        option,
        loop='3D',
        block_size=(32, 8, 1),
        halo=1,
    )

    stream_kwargs = kwargs.copy()
    stream_kwargs.update(loop='1D', block_size=(1024, 1, 1), halo=0)

    configurations = [
        Configuration(basic.Copy, name='stream', **stream_kwargs),
        Configuration(basic.Empty, name='empty', **kwargs),
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

    table = run_scaling_benchmark(configurations, executions)
    table.to_csv(output)


@main.command()
@click.argument('output', type=click.Path())
@click.option('--executions', '-e', type=int, default=101)
@click.option('--option', '-o', multiple=True)
def horizontal_diffusion_bandwidth(output, executions, option):
    kwargs = common_kwargs(option)

    configurations = [
        Configuration(hdiff.Classic, block_size=(32, 16, 1), **kwargs),
        Configuration(hdiff.OnTheFly,
                      block_size=(256, 2, 1),
                      loop='3D',
                      **kwargs),
        Configuration(hdiff.OnTheFlyIncache, block_size=(32, 8, 1), **kwargs),
        Configuration(hdiff.JScanSharedMem, block_size=(256, 32, 1), **kwargs),
        Configuration(hdiff.JScanOtfIncache, block_size=(256, 4, 1), **kwargs),
        Configuration(hdiff.JScanOtf, block_size=(256, 2, 1), **kwargs),
        Configuration(hdiff.JScanShuffleIncache,
                      block_size=(28, 4, 2),
                      **kwargs),
        Configuration(hdiff.JScanShuffle, block_size=(28, 3, 2), **kwargs),
        Configuration(hdiff.JScanShuffleSystolic,
                      block_size=(28, 2, 2),
                      **kwargs)
    ]

    def truncate_block_size_to_domain_if_possible(**kwargs):
        if kwargs['block_size'][0] != 28:
            return truncate_block_size_to_domain(**kwargs)
        return kwargs

    table = run_scaling_benchmark(
        configurations,
        executions,
        preprocess_args=truncate_block_size_to_domain_if_possible)
    table.to_csv(output)


@main.command()
@click.argument('output', type=click.Path())
@click.option('--executions', '-e', type=int, default=101)
@click.option('--option', '-o', multiple=True)
def vertical_advection_bandwidth(output, executions, option):
    kwargs = common_kwargs(option)

    configurations = [
        Configuration(vadv.Classic,
                      block_size=(512, 2),
                      unroll_factor=5,
                      **kwargs),
        Configuration(vadv.LocalMem,
                      block_size=(64, 4),
                      unroll_factor=8,
                      **kwargs),
        Configuration(vadv.SharedMem,
                      block_size=(64, 1),
                      unroll_factor=0,
                      **kwargs),
        Configuration(vadv.LocalMemMerged,
                      block_size=(32, 1),
                      unroll_factor=0,
                      **kwargs)
    ]

    table = run_scaling_benchmark(
        configurations,
        executions,
        preprocess_args=truncate_block_size_to_domain)
    table.to_csv(output)


if __name__ == '__main__':
    main()
