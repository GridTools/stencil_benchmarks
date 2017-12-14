#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import collections
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

def argsdiff(argslist):
    diffkeys = set()
    first = argslist[0]
    for a in argslist[1:]:
        assert a.keys() == first.keys()
        for k, v in first.items():
            if a[k] != v:
                diffkeys.add(k)
    return set(diffkeys)

def plot_title(args):
    if isinstance(args, dict):
        diff = set()
    else:
        diff = argsdiff(args)
        args = args[0]
    title = []
    if 'variant' not in diff:
        title.append(u'Variant: {}'.format(args['variant']))
    if 'stencil' not in diff:
        title.append(u'Stencil: {}'.format(args['stencil']))
    if 'prec' not in diff:
        title.append(u'Prec: {}'.format(args['precision'].title()))
    if 'align' not in diff:
        title.append(u'Align: {}'.format(args['alignment']))
    if 'threads' not in diff and int(args['threads']) != 0:
        title.append(u'Threads: {}'.format(args['threads']))
    if diff.isdisjoint({'i-size', 'j-size', 'k-size'}):
        title.append(u'Domain: {}×{}×{}'.format(args['i-size'], args['j-size'], args['k-size']))
    if 'halo' not in diff:
        title.append(u'Halo: {}'.format(args['halo']))
    if diff.isdisjoint({'i-layout', 'j-layout', 'k-layout'}):
        title.append(u'Layout: {}-{}-{}'.format(args['i-layout'], args['j-layout'], args['k-layout']))

    splittitle, minlen = None, 1000000
    for s in range(len(title)):
        ts = u' — '.join(title[:s]), u' — '.join(title[s:])
        maxlen = max(len(t) for t in ts)
        if maxlen < minlen:
            splittitle, minlen = u'\n'.join(ts), maxlen
    return splittitle

def metric_str(args):
    if args['metric'].lower() == 'time':
        return 'Measured Time [ms]'
    elif args['metric'].lower() == 'bandwidth':
        return 'Estimated Bandwidth [GB/s]'
    elif args['metric'].lower() == 'papi':
        return args['papi-event']
    elif args['metric'].lower() == 'papi-imbalance':
        return 'Imbalance of ' + args['papi-event']

def metric_abbr(args):
    if args['metric'].lower() == 'time':
        return 'Time'
    elif args['metric'].lower() == 'bandwidth':
        return 'BW'
    elif args['metric'].lower() == 'papi':
        return 'CTR'
    elif args['metric'].lower() == 'papi-imbalance':
        return 'CTR-IMB'

def plot_single_size(args, data, logscale=False, lim=None):
    assert args['run-mode'] == 'single-size'

    x = np.arange(len(data.index))
    m = metric_abbr(args)
    mavg = data[m + '-avg'].values
    mmin = data[m + '-min'].values
    mmax = data[m + '-max'].values
    plt.bar(x, mavg, 0.6, yerr=[mavg - mmin, mmax - mavg])
    plt.xticks(x, rotation=45)
    plt.gca().set_xticklabels(data.index)
    plt.grid(axis='y')
    plt.gca().set_axisbelow(True)
    plt.xlabel('Stencil')
    plt.ylabel(metric_str(args))
    if lim:
        plt.ylim(lim)

def plot_ij_scaling(args, data, logscale=False, lim=None):
    assert args['run-mode'] == 'ij-scaling'
    assert args['i-size'] == args['j-size']

    if logscale and np.amax(data.values) <= 0:
        logscale = False

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

    if lim is not None:
        vmin, vmax = lim
    else:
        vmin = np.amin(data.values)
        vmax = np.amax(data.values)

    if vmin <= 0 or vmax - vmin == 0:
        logscale = False

    mul = 1
    if viewscale:
        vabsmax = max(abs(vmin), abs(vmax))
        if vabsmax != 0:
            while round(vabsmax / mul) >= 10000:
                mul *= 10
            while round(vabsmax / mul) < 1000:
                mul /= 10.0
    assert mul > 0

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

def plot_blocksize_reduction(allargs, alldata, diffkeys, op, logscale=False, lim=None):
    x = np.arange(len(allargs))
    m = metric_abbr(allargs[0]) 

    if op is None:
        mavg = [np.mean(data.values) for data in alldata]
        mmin = [np.amin(data.values) for data in alldata]
        mmax = [np.amax(data.values) for data in alldata]
        plt.fill_between(x, mmin, mmax, alpha=0.5)
        plt.plot(x, mavg)
    elif op == 'avg':
        plt.plot(x, [np.mean(data.values) for data in alldata])
    elif op == 'min':
        plt.plot(x, [np.amin(data.values) for data in alldata])
    elif op == 'max':
        plt.plot(x, [np.amax(data.values) for data in alldata])
    else:
        assert False

    if logscale:
        plt.yscale('log')

    plt.xticks(x, rotation=45)
    if 'i-size' in diffkeys or 'j-size' in diffkeys or 'k-size' in diffkeys:
        isizes = [int(a['i-size']) for a in allargs]
        jsizes = [int(a['j-size']) for a in allargs]
        ksizes = [int(a['k-size']) for a in allargs]
        xlabels = ['{}x{}x{}'.format(i, j, k) for i, j, k in zip(isizes, jsizes, ksizes)]

    plt.gca().set_xticklabels(xlabels)
    plt.grid()
    plt.gca().set_axisbelow(True)
    plt.ylabel(metric_str(allargs[0]))
    if lim:
        plt.ylim(lim)

@click.group()
@click.pass_context
def cli(ctx):
    matplotlib.rcParams.update({'font.size': 25,
                                'xtick.labelsize': 'small',
                                'ytick.labelsize': 'small',
                                'axes.titlesize': 'small',
                                'lines.linestyle': '--',
                                'lines.linewidth': 2})

    pass

@cli.command()
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
def single(outfile, infile, logscale, vmin, vmax, viewscale):
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

        if args['run-mode'] == 'single-size':
            plot_single_size(args, data, logscale, lim)
        if args['run-mode'] == 'ij-scaling':
            plot_ij_scaling(args, data, logscale, lim)
        if args['run-mode'] == 'blocksize-scan':
            plot_blocksize_scan(args, data, logscale, lim, viewscale)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

@cli.command()
@click.argument('outfile', type=click.Path())
@click.argument('infile', type=click.Path(exists=True), nargs=-1)
@click.option('--logscale/--no-logscale', default=False,
              help='Use logarithmic scaling for dependent variable.')
@click.option('--vmin', help='Minimum value of dependent variable used to define visible data range', type=float)
@click.option('--vmax', help='Maximum value of dependent variable used to define visible data range', type=float)
@click.option('--local-reduction', help='Per-file reduction operation', type=click.Choice(['min', 'max', 'avg']))
def reduce(outfile, infile, logscale, vmin, vmax, local_reduction):
    allargs, alldata = zip(*(read(f) for f in infile))
    diff = argsdiff(allargs)
    diff.remove('output')
    plt.figure(figsize=(12, 10))
    plt.title(plot_title(allargs))
    if allargs[0]['run-mode'] == 'blocksize-scan':
        assert diff <= {'i-size', 'j-size', 'k-size'}
        plot_blocksize_reduction(allargs, alldata, diff, local_reduction, logscale, [vmin, vmax])
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

if __name__ == '__main__':
    cli()

