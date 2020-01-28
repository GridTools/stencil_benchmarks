#!/usr/bin/env python

import math

import click

from stencil_benchmarks.benchmarks_collection.stencils.openmp import basic
from stencil_benchmarks.tools.multirun import (Configuration,
                                               run_scaling_benchmark)
from stencil_benchmarks.tools import alloc


@click.group()
def main():
    pass


def common_kwargs(options=None, **kwargs):
    kwargs = dict(platform_preset='none',
                  alignment=alloc.l1_dcache_linesize(),
                  **kwargs)
    for o in options:
        name, value = o.split('=')
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
@click.option('--domain', type=int, nargs=3, default=(1024, 1024, 80))
def blocked_copy(output, executions, dtype, option, domain):
    kwargs = common_kwargs(option, dtype=dtype, loop='3D', halo=1)

    stream_kwargs = kwargs.copy()
    stream_kwargs.update(loop='1D', halo=0)

    configurations = [
        Configuration(basic.Copy, name='stream', **stream_kwargs),
    ]

    def blocks(d):
        for i in range(int(math.log2(d))):
            yield 2**i

    for bx in blocks(domain[0]):
        for by in blocks(domain[1]):
            for bz in blocks(domain[2]):
                configurations.append(
                    Configuration(basic.Copy,
                                  name=f'blocked-{bx}-{by}-{bz}',
                                  block_size=(bx, by, bz),
                                  **kwargs))

    table = run_scaling_benchmark(configurations,
                                  executions,
                                  domain_range=[domain])
    table.to_csv(output)


if __name__ == '__main__':
    main()
