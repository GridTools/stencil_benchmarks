#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import itertools
import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def read_header(filename):
    args = dict()
    with open(filename, 'r') as f:
        for line in f:
            l = line.split()
            if l[0] == '#':
                for k, v in zip(l[1::2], l[2::2]):
                    if k[-1] == ':':
                        args[k[:-1]] = v
    return args

def read_data(filename):
    return pd.read_csv(filename, comment='#', sep='\s+', index_col=0)

def read(filename):
    return read_header(filename), read_data(filename)

def plot_title(args):
    return (u'Variant: {} — Stencil: {} — Prec: {} — Align: {}\n'
            u'Threads: {} — Domain: {}×{}×{} — Halo: {} — Layout: {}-{}-{}').format(
                    args['platform'] + ' ' + args['variant'], args['stencil'],
                    args['precision'].title(), args['alignment'],
                    args['threads'],
                    args['i-size'], args['j-size'], args['k-size'],
                    args['halo'],
                    args['i-layout'], args['j-layout'], args['k-layout'])

def metric_str(args):
    if args['metric'] == 'time':
        return 'Measured Time [s]'
    elif args['metric'] == 'bandwidth':
        return 'Estimated Bandwidth [GB/s]'
    elif args['metric'] == 'PAPI':
        return args['papi-event']

def plot_ij_scaling(args, data, logscale=False, lim=None):
    assert args['run-mode'] == 'ij-scaling'
    assert args['i-size'] == args['j-size']

    x = np.array(data.columns, dtype=int) + 2 * int(args['halo'])
    for row in data.itertuples(name=None):
        stencil, bw = row[0], row[1:]
        if logscale:
            plt.loglog(x, bw, basex=2, lw=2, ls='--', label=stencil)
        else:
            plt.semilogx(x, bw, basex=2, lw=2, ls='--', label=stencil)
    mstr = metric_str(args)
    plt.xlabel('Domain Size (Including Halo)')
    plt.ylabel(mstr)
    plt.xticks(x)
    if lim:
        plt.ylim(lim)
    plt.xlim([np.amin(x), np.amax(x)])
    plt.gca().xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda s, _: '{0}x{0}x{1}'.format(int(s),
                int(args['k-size']) + 2 * int(args['halo']))))
    plt.grid()
    plt.legend()

def plot_blocksize_scan(args, data, logscale=False, lim=None, viewscale=True):
    assert args['run-mode'] == 'blocksize-scan'

    mul = 1
    if viewscale:
        vmax = lim[1] if lim is not None else np.amax(data.values)
        vmax = int(round(vmax))
        while vmax >= 10000 * mul:
            mul *= 10
        while vmax < 1000 * mul:
            mul /= 10.0

    mstr = metric_str(args)
    imargs = dict()
    if lim:
        imargs['vmin'], imargs['vmax'] = lim[0] / mul, lim[1] / mul
    if logscale:
        imargs['norm'] = matplotlib.colors.LogNorm()
    plt.imshow(data.values.T / mul, origin='lower', interpolation='nearest', **imargs)
    x = np.array(data.index, dtype=int)
    y = np.array(data.columns, dtype=int)
    plt.xticks(np.arange(x.size), x)
    plt.yticks(np.arange(y.size), y)
    plt.xlabel('i-Blocksize')
    plt.ylabel('j-Blocksize')
    if mul == 1:
        mulstr = ''
    elif mul > 1:
        mulstr = u' ∕ {}'.format(int(mul))
    else:
        mulstr = u' x {}'.format(int(1 / mul))
    cbar = plt.colorbar(label=mstr + mulstr,
                        fraction=0.046, pad=0.04)
    if logscale:
        if lim:
            ticks = np.logspace(np.log10(lim[0] / mul),
                                np.log10(lim[1] / mul), 5)
        else:
            ticks = np.logspace(np.log10(int(round(np.amin(data.values) / mul))),
                                np.log10(int(round(np.amax(data.values) / mul))), 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([int(round(t)) for t in ticks])
    for i in range(x.size):
        for j in range(y.size):
            v = data.values[i, j]
            plt.text(i, j, '{}'.format(int(round(data.values[i, j] / mul))),
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize='xx-small', color='white')


@click.command()
@click.argument('outfile', type=click.Path())
@click.argument('infile', type=click.Path(exists=True), nargs=-1)
@click.option('--logscale/--no-logscale', default=False,
              help='Use logarithmic scaling for dependent variable.')
@click.option('--vmin', metavar='[FLOAT|common]',
              help='Minimum value of dependent variable used to define visible data range or "common" to use the minimum data value of all input files.')
@click.option('--vmax', metavar='[FLOAT|common]',
              help='Maximum value of dependent variable used to define visible data range or "common" to use the maximum data value of all input files.')
@click.option('--viewscale/--no-viewscale', default=True,
              help='Automatically scale numbers by a multiplier, for well readable value range.')
def cli(outfile, infile, logscale, vmin, vmax, viewscale):
    matplotlib.rcParams.update({'font.size': 25,
                                'xtick.labelsize': 'small',
                                'ytick.labelsize': 'small',
                                'axes.titlesize': 'small'})

    nrows = int(np.sqrt(len(infile)))
    ncols = (len(infile) + nrows - 1) // nrows

    indata = [read(f) for f in infile]

    if vmin is None and vmax is None:
        lim = None
    else:
        if vmin is None or vmin == 'common':
            vmin = min(np.amin(data.values) for _, data in indata)
        else:
            vmin = float(vmin)
        if vmax is None or vmax == 'common':
            vmax = max(np.amax(data.values) for _, data in indata)
        else:
            vmax = float(vmax)
        lim = [vmin, vmax]

    plt.figure(figsize=(12 * ncols, 10 * nrows))
    for i, d in enumerate(indata):
        plt.subplot(nrows, ncols, i + 1)
        args, data = d
        plt.title(plot_title(args), y=1.05)

        if args['run-mode'] == 'ij-scaling':
            plot_ij_scaling(args, data, logscale, lim)
        if args['run-mode'] == 'blocksize-scan':
            plot_blocksize_scan(args, data, logscale, lim, viewscale)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

if __name__ == '__main__':
    cli()

