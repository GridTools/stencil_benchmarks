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

def plot_ij_scaling(args, data):
    assert args['run-mode'] == 'ij-scaling'
    assert args['i-size'] == args['j-size']

    x = np.array(data.columns, dtype=int) + 2 * int(args['halo'])
    for row in data.itertuples(name=None):
        stencil, bw = row[0], row[1:]
        plt.semilogx(x, bw, basex=2, lw=2, ls='--', label=stencil)
    plt.xlabel('Domain Size (Including Halo)')
    plt.ylabel('Estimated Bandwidth [GB/s]')
    plt.xticks(x)
    plt.ylim([0, 500])
    plt.xlim([np.amin(x), np.amax(x)])
    plt.gca().xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda s, _: '{0}x{0}x{1}'.format(int(s),
                int(args['k-size']) + 2 * int(args['halo']))))
    plt.grid()
    plt.legend()

def plot_blocksize_scan(args, data):
    assert args['run-mode'] == 'blocksize-scan'

    plt.imshow(data.values.T, origin='lower', vmin=0, vmax=500)
    x = np.array(data.index, dtype=int)
    y = np.array(data.columns, dtype=int)
    plt.xticks(np.arange(x.size), x)
    plt.yticks(np.arange(y.size), y)
    plt.xlabel('i-Blocksize')
    plt.ylabel('j-Blocksize')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Estimated Bandwidth [GB/s]')
    for i in range(x.size):
        for j in range(y.size):
            plt.text(i, j, '{:.0f}'.format(data.values[i, j]),
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize='x-small', color='white')


@click.command()
@click.argument('outfile', type=click.Path())
@click.argument('infile', type=click.Path(exists=True), nargs=-1)
def cli(outfile, infile):
    matplotlib.rcParams.update({'font.size': 25,
                                'xtick.labelsize': 'small',
                                'ytick.labelsize': 'small',
                                'axes.titlesize': 'small'})

    nrows = int(np.sqrt(len(infile)))
    ncols = (len(infile) + nrows - 1) // nrows

    plt.figure(figsize=(12 * ncols, 10 * nrows))
    for i, f in enumerate(infile):
        plt.subplot(nrows, ncols, i + 1)
        args, data = read(f)
        plt.title(plot_title(args), y=1.05)

        if args['run-mode'] == 'ij-scaling':
            plot_ij_scaling(args, data)
        if args['run-mode'] == 'blocksize-scan':
            plot_blocksize_scan(args, data)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

if __name__ == '__main__':
    cli()

