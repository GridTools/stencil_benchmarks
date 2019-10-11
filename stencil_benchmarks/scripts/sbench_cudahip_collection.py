#!/usr/bin/env python

import re

import click
import pandas as pd

from stencil_benchmarks.benchmarks_collection.stencils.cuda_hip import (
    basic, horizontal_diffusion as hdiff, vertical_advection as vadv)
from stencil_benchmarks.tools import cli


class Configuration:
    def __init__(self, ctor, name=None, **kwargs):
        self.ctor = ctor
        if name is None:
            self.name = re.sub('(?!^)([A-Z1-9]+)', r'-\1',
                               ctor.__name__).lower()
        else:
            self.name = name
        self.kwargs = kwargs

    def __call__(self, preprocess_args=None, **kwargs):
        if preprocess_args is None:

            def preprocess_args(**kwargs):
                return kwargs

        run = self.ctor(**preprocess_args(**self.kwargs, **kwargs))

        def modified_run():
            result = run()
            result.update(name=self.name)
            result.update(cli.pretty_parameters(run))
            return result

        return modified_run


def benchmark_domains():
    for exponent in range(5, 11):
        d = 2**exponent
        yield d, d, 80


def truncate_block_size_to_domain(**kwargs):
    if 'block_size' in kwargs and 'domain' in kwargs:
        kwargs['block_size'] = tuple(
            min(b, d) for b, d in zip(kwargs['block_size'], kwargs['domain']))
    return kwargs


def benchmark(configurations, executions, preprocess_args=None):
    results = []
    with cli.ProgressBar() as progress:
        for domain in progress.report(benchmark_domains()):
            for config in progress.report(configurations):
                run = config(preprocess_args=preprocess_args, domain=domain)
                results += [run() for _ in progress.report(range(executions))]
    return pd.DataFrame(results)


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

    table = benchmark(configurations, executions)
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

    table = benchmark(
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

    table = benchmark(configurations,
                      executions,
                      preprocess_args=truncate_block_size_to_domain)
    table.to_csv(output)


if __name__ == '__main__':
    main()
