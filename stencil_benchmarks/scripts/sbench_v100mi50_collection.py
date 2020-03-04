#!/usr/bin/env python

import click

from stencil_benchmarks.benchmarks_collection.stencils.cuda_hip import (
    basic, horizontal_diffusion as hdiff, vertical_advection as vadv)
from stencil_benchmarks.tools.multirun import (Configuration,
                                               run_scaling_benchmark,
                                               truncate_block_size_to_domain)


@click.group()
def main():
    pass


def common_kwargs(backend, gpu_architecture, dtype):
    return dict(backend=backend,
                compiler='nvcc' if backend == 'cuda' else 'hipcc',
                gpu_architecture=gpu_architecture,
                verify=False,
                run_twice=True,
                gpu_timers=True,
                alignment=128 if backend == 'cuda' else 64,
                dtype=dtype)


@main.command()
@click.argument('backend', type=click.Choice(['cuda', 'hip']))
@click.argument('gpu-architecture')
@click.argument('output', type=click.Path())
@click.option('--executions', '-e', type=int, default=101)
@click.option('--dtype', '-d', default='float32')
def basic_bandwidth(backend, gpu_architecture, output, executions, dtype):
    kwargs = common_kwargs(backend, gpu_architecture, dtype)
    kwargs.update(
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
@click.argument('backend', type=click.Choice(['cuda', 'hip']))
@click.argument('gpu-architecture')
@click.argument('output', type=click.Path())
@click.option('--executions', '-e', type=int, default=101)
@click.option('--dtype', '-d', default='float32')
def horizontal_diffusion_bandwidth(backend, gpu_architecture, output,
                                   executions, dtype):
    kwargs = common_kwargs(backend, gpu_architecture, dtype)

    def choose(hip, cuda):
        return hip if backend == 'hip' else cuda

    configurations = [
        Configuration(hdiff.Classic,
                      block_size=choose((64, 8, 1), (32, 16, 1)),
                      **kwargs),
        Configuration(hdiff.OnTheFly,
                      block_size=choose((128, 4, 1), (256, 2, 1)),
                      loop='3D',
                      **kwargs),
        Configuration(hdiff.OnTheFlyIncache,
                      block_size=choose((32, 8, 4), (32, 8, 1)),
                      **kwargs),
        Configuration(hdiff.JScanSharedMem,
                      block_size=choose((512, 16, 1), (256, 32, 1)),
                      **kwargs),
        Configuration(hdiff.JScanOtfIncache,
                      block_size=choose((128, 4, 1), (256, 4, 1)),
                      **kwargs),
        Configuration(hdiff.JScanOtf,
                      block_size=choose((256, 4, 1), (256, 2, 1)),
                      **kwargs),
        Configuration(hdiff.JScanShuffleIncache,
                      block_size=choose((60, 4, 1), (28, 4, 2)),
                      **kwargs),
        Configuration(hdiff.JScanShuffle,
                      block_size=choose((60, 3, 1), (28, 3, 2)),
                      **kwargs),
        Configuration(hdiff.JScanShuffleSystolic,
                      block_size=choose((60, 4, 1), (28, 2, 2)),
                      **kwargs)
    ]

    def truncate_block_size_to_domain_if_possible(**kwargs):
        if kwargs['block_size'][0] not in (28, 60):
            return truncate_block_size_to_domain(**kwargs)
        return kwargs

    table = run_scaling_benchmark(
        configurations,
        executions,
        preprocess_args=truncate_block_size_to_domain_if_possible)
    table.to_csv(output)


@main.command()
@click.argument('backend', type=click.Choice(['cuda', 'hip']))
@click.argument('gpu-architecture')
@click.argument('output', type=click.Path())
@click.option('--executions', '-e', type=int, default=101)
@click.option('--dtype', '-d', default='float32')
def vertical_advection_bandwidth(backend, gpu_architecture, output, executions,
                                 dtype):
    kwargs = common_kwargs(backend, gpu_architecture, dtype)

    def choose(hip, cuda):
        return hip if backend == 'hip' else cuda

    configurations = [
        Configuration(vadv.Classic,
                      block_size=choose((128, 8), (512, 2)),
                      unroll_factor=choose(3, 5),
                      **kwargs),
        Configuration(vadv.LocalMem,
                      block_size=choose((256, 1), (64, 4)),
                      unroll_factor=choose(10, 8),
                      **kwargs),
        Configuration(vadv.SharedMem,
                      block_size=choose((32, 1), (64, 1)),
                      unroll_factor=choose(6, 0),
                      **kwargs)
    ]

    if backend == 'cuda':
        # does not verify on HIPCC (compiler bug?)
        configurations += [
            Configuration(vadv.LocalMemMerged,
                          block_size=(32, 1),
                          unroll_factor=choose(-1, 0),
                          **kwargs)
        ]

    table = run_scaling_benchmark(
        configurations,
        executions,
        preprocess_args=truncate_block_size_to_domain)
    table.to_csv(output)


if __name__ == '__main__':
    main()
