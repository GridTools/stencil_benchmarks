#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def read(filenames):
    dfs = []
    for f in filenames:
        try:
            dfs.append(pd.read_csv(f, sep='\s+'))
        except pd.errors.EmptyDataError:
            pass
    return pd.concat(dfs)


@click.group()
def cli():
    matplotlib.rcParams.update({'font.size': 25,
                                'xtick.labelsize': 'small',
                                'ytick.labelsize': 'small',
                                'axes.titlesize': 'small',
                                'lines.linestyle': '-',
                                'lines.marker': 'o',
                                'lines.linewidth': 3})


@cli.command()
@click.argument('x')
@click.argument('y')
@click.argument('outfile', type=click.Path())
@click.argument('infile', type=click.Path(exists=True), nargs=-1)
@click.option('--reduction', '-r', type=click.Choice(['min', 'max', 'avg']), default='avg')
@click.option('--label', '-l')
@click.option('--subplot', '-s')
@click.option('--xlog', is_flag=True)
@click.option('--ylog', is_flag=True)
def line(x, y, infile, outfile, reduction, label, subplot, xlog, ylog):

    def red(grpd):
        if reduction == 'min':
            return grpd.min()
        elif reduction == 'max':
            return grpd.max()
        elif reduction =='avg':
            return grpd.mean()
        else:
            raise ValueError()

    def plot(df):
        if label:
            lvs = df[label].unique()
            for lv in lvs:
                ldf = df[df[label] == lv]
                data = red(ldf.groupby(x))
                plt.plot(data.index, data[y], label=lv)
            plt.legend()
        else:
            data = red(df.groupby(x))
            plt.plot(data.index, data[y], label='lv')
        plt.xlabel(x.title())
        plt.ylabel(y.title())
        plt.grid(which='major', alpha=0.5)
        plt.grid(which='minor', alpha=0.2)
        plt.minorticks_on()
        if xlog:
            plt.gca().set_xscale('log')
        if ylog:
            plt.gca().set_yscale('log')
        else:
            plt.gca().set_ylim(bottom=0)

    plt.figure(figsize=(12, 10))

    df = read(infile)
    if subplot:
        sbs = df[subplot].unique()
        nrows = math.floor(math.sqrt(len(sbs)))
        ncols = math.ceil(len(sbs) / float(nrows))
        for i, sb in enumerate(sbs):
            plt.subplot(nrows, ncols, i + 1)
            plt.title(sb.title())
            plot(df[df[subplot] == sb])
    else:
        plot(df)

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


if __name__ == '__main__':
    cli()
