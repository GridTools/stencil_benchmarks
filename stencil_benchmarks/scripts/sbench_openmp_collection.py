#!/usr/bin/env python

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
    kwargs = dict(platform_preset='none',
                  alignment=alloc.l1_dcache_linesize(),
                  verify=False,
                  huge_pages=True)
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


def scale_domain(**kwargs):
    i, j, k = kwargs['domain']
    kwargs['domain'] = (i // 2, j // 2, k)
    return truncate_block_size_to_domain(**kwargs)


@main.command()
@click.argument('output', type=click.Path())
@click.option('--executions', '-e', type=int, default=101)
@click.option('--dtype', '-d', default='float32')
@click.option('--option', '-o', multiple=True)
def basic_bandwidth(output, executions, dtype, option):
    kwargs = common_kwargs(option,
                           dtype=dtype,
                           loop='3D-blocked',
                           halo=1,
                           block_size=(1024, 16, 1))

    stream_kwargs = kwargs.copy()
    stream_kwargs.update(loop='1D', halo=0)

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
    table = run_scaling_benchmark(configurations,
                                  executions,
                                  preprocess_args=scale_domain)
    table.to_csv(output)


@main.command()
@click.argument('output', type=click.Path())
@click.option('--executions', '-e', type=int, default=101)
@click.option('--dtype', '-d', default='float32')
@click.option('--option', '-o', multiple=True)
def horizontal_diffusion_bandwidth(output, executions, dtype, option):
    vector_size = 64 // np.dtype(dtype).itemsize
    kwargs = common_kwargs(option,
                           dtype=dtype,
                           alignment=64,
                           vector_size=vector_size)

    configurations = [
        Configuration(hdiff.ClassicVec, **kwargs, block_size=(1024, 16, 1)),
        Configuration(hdiff.OnTheFlyVec, **kwargs, block_size=(1024, 8, 1)),
        Configuration(hdiff.MinimumMem, **kwargs, block_size=(1024, 64, 1))
    ]

    table = run_scaling_benchmark(configurations,
                                  executions,
                                  preprocess_args=scale_domain)
    table.to_csv(output)


@main.command()
@click.argument('output', type=click.Path())
@click.option('--executions', '-e', type=int, default=101)
@click.option('--dtype', '-d', default='float32')
@click.option('--option', '-o', multiple=True)
def vertical_advection_bandwidth(output, executions, dtype, option):
    vector_size = 64 // np.dtype(dtype).itemsize
    kwargs = common_kwargs(option,
                           dtype=dtype,
                           layout=(2, 0, 1),
                           vector_size=vector_size)

    configurations = [
        Configuration(vadv.KMiddleVec, **kwargs, block_size=(1024, 1)),
        Configuration(vadv.KInnermostVec,
                      **kwargs,
                      block_size=(64, 1),
                      prefetch_distance=4),
        Configuration(vadv.KInnermostBlockVec,
                      **kwargs,
                      block_size=(64, 1),
                      prefetch_distance=2)
    ]

    table = run_scaling_benchmark(configurations,
                                  executions,
                                  preprocess_args=scale_domain)
    table.to_csv(output)


if __name__ == '__main__':
    main()
