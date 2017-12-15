#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def figsize(ncols=1, nrows=1):
    return base_figsize[0] * ncols, base_figsize[1] * nrows


def read(filenames):
    dfs = []
    for f in filenames:
        try:
            dfs.append(pd.read_csv(f, sep='\s+'))
        except pd.errors.EmptyDataError:
            pass
    return pd.concat(dfs)


@click.group()
@click.option('--width', '-w', type=float, default=16)
@click.option('--height', '-h', type=float, default=10)
def cli(width, height):
    matplotlib.rcParams.update({'font.size': 25,
                                'xtick.labelsize': 'small',
                                'ytick.labelsize': 'small',
                                'axes.titlesize': 'small',
                                'lines.linestyle': '-',
                                'lines.marker': 'o',
                                'lines.linewidth': 3})
    global base_figsize
    base_figsize = (width, height)


def red(grpd, reduction):
    if reduction == 'min':
        return grpd.min()
    elif reduction == 'max':
        return grpd.max()
    elif reduction == 'avg':
        return grpd.mean()
    else:
        raise ValueError()


@cli.command()
@click.argument('x')
@click.argument('y')
@click.argument('outfile', type=click.Path())
@click.argument('infile', type=click.Path(exists=True), nargs=-1)
@click.option('--reduction', '-r', type=click.Choice(['min', 'max', 'avg']),
              default='avg')
@click.option('--label', '-l')
@click.option('--subplot', '-s')
@click.option('--xscale', type=click.Choice(['linear', 'log', 'log2']),
              default='linear')
@click.option('--yscale', type=click.Choice(['linear', 'log', 'log2']),
              default='linear')
@click.option('--legend/--no-legend', default=True)
@click.option('--data-xticks/--no-data-xticks', default=False)
def line(x, y, infile, outfile, reduction, label, subplot, xscale, yscale,
         legend, data_xticks):

    def plot(df):
        xticks = set()
        if label:
            lvs = df[label].unique()[::-1]
            for lv in lvs:
                ldf = df[df[label] == lv]
                data = red(ldf.groupby(x), reduction)
                plt.plot(data.index, data[y], label=lv)
                print(' '.join('{:10.5f}'.format(d) for d in data.index))
                print(' '.join('{:10.5f}'.format(d) for d in data[y]))
                xticks = xticks | set(data.index)
            if legend:
                plt.legend()
        else:
            data = red(df.groupby(x), reduction)
            plt.plot(data.index, data[y], label='lv')
            print(' '.join('{:10.5f}'.format(d) for d in data.index))
            print(' '.join('{:10.5f}'.format(d) for d in data[y]))
            xticks = xticks | set(data.index)
        plt.xlabel(str(x).title())
        plt.ylabel(str(y).title())
        plt.grid(which='major', alpha=0.5)
        plt.grid(which='minor', alpha=0.2)
        plt.minorticks_on()
        if xscale == 'log':
            plt.gca().set_xscale('log', basex=10)
        elif xscale == 'log2':
            plt.gca().set_xscale('log', basex=2)
        if yscale == 'log':
            plt.gca().set_yscale('log', basey=10)
        elif yscale == 'log2':
            plt.gca().set_yscale('log', basey=2)
        else:
            plt.gca().set_ylim(bottom=0)
        if data_xticks:
            xticks = sorted(list(xticks))
            plt.xticks(xticks, xticks)

    df = read(infile)
    if subplot:
        sbs = df[subplot].unique()
        nrows = math.floor(math.sqrt(len(sbs)))
        ncols = math.ceil(len(sbs) / float(nrows))
        plt.figure(figsize=figsize(ncols, nrows))
        for i, sb in enumerate(sbs):
            plt.subplot(nrows, ncols, i + 1)
            plt.title(str(sb).title())
            plot(df[df[subplot] == sb])
    else:
        plt.figure(figsize=figsize())
        plot(df)

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()


@cli.command()
@click.argument('x')
@click.argument('y')
@click.argument('z')
@click.argument('outfile', type=click.Path())
@click.argument('infile', type=click.Path(exists=True), nargs=-1)
@click.option('--reduction', '-r', type=click.Choice(['min', 'max', 'avg']),
              default='avg')
@click.option('--subplot', '-s')
@click.option('--viewscale', '-v', is_flag=True)
def heatmap(x, y, z, infile, outfile, reduction, subplot, viewscale):

    def plot(df):
        data = red(df.groupby([x, y]), reduction)
        data = data.pivot_table(values=z, index=x, columns=y)

        mul = 1
        if viewscale:
            vmin = np.nanmin(data.values)
            vmax = np.nanmax(data.values)
            vabsmax = max(abs(vmin), abs(vmax))
            if vabsmax != 0:
                while round(vabsmax / mul) >= 10000:
                    mul *= 10
                while round(vabsmax / mul) < 1000:
                    mul /= 10.0
            assert mul > 0

        plt.imshow(data.values.T / mul, origin='lower',
                   interpolation='nearest')

        if mul == 1:
            mulstr = ''
        elif mul > 1:
            mulstr = u' ∕ {}'.format(int(mul))
        else:
            mulstr = u' x {}'.format(int(1 / mul))

        plt.colorbar(label=str(z).title() + mulstr)
        plt.xlabel(str(x).title())
        plt.ylabel(str(y).title())
        xticks = np.array(data.index)
        yticks = np.array(data.columns)
        plt.xticks(np.arange(xticks.size), xticks)
        plt.yticks(np.arange(yticks.size), yticks)
        for i in range(xticks.size):
            for j in range(yticks.size):
                try:
                    vstr = '{}'.format(int(round(data.values[i, j] / mul)))
                except ValueError:
                    vstr = 'NaN'
                plt.text(i, j, vstr,
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize='xx-small', color='white')

    df = read(infile)
    if subplot:
        sbs = np.sort(df[subplot].unique())
        nrows = math.floor(math.sqrt(len(sbs)))
        ncols = math.ceil(len(sbs) / float(nrows))
        plt.figure(figsize=figsize(ncols, nrows))
        for i, sb in enumerate(sbs):
            plt.subplot(nrows, ncols, i + 1)
            plt.title(str(subplot).title() + ' = ' + str(sb).title())
            plot(df[df[subplot] == sb])
    else:
        plt.figure(figsize=figsize())
        plot(df)

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    cli()
