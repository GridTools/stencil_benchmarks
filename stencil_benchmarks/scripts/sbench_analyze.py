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

import functools
import itertools
import operator
import os
import re
import textwrap

import click
import numpy as np
import pandas as pd


@click.group()
def main():
    pass


def read_csv(csv):
    return pd.read_csv(csv,
                       index_col=0,
                       na_values=['nan', 'NaN'],
                       keep_default_na=False)


def strip(df, invert=False):
    nunique = df.apply(pd.Series.nunique)
    which = nunique == 1 if not invert else nunique > 1
    df = df.drop(nunique[which].index, axis='columns')
    if invert:
        df = df.iloc[0].sort_index()
    return df


@main.command(name='print')
@click.argument('csv', type=click.Path(exists=True))
@click.option(
    '--common/--non-common',
    '-c',
    default=False,
    help='Only print values common for all sbench runs in the input file.')
@click.option('--auto-group/--no-auto-group',
              '-a',
              default=False,
              help='Automatically group values.')
@click.option('--group',
              '-g',
              multiple=True,
              help='Group values by data column with the given name.')
@click.option('--select',
              '-s',
              multiple=True,
              help='Only print values from selected columns.')
@click.option('--pivot',
              '-p',
              nargs=2,
              help='Create a pivot table with the given columns as axes.')
@click.option('--aggregation',
              default='median',
              help='Aggregation function used for pivotting.')
@click.option('--sort/--no-sort', default=False, help='Sort values.')
@click.option('--output', '-o', type=click.File(mode='w'), help='Output file.')
def print_csv(csv, common, auto_group, group, select, pivot, aggregation, sort,
              output):
    """Print reports of CSV files produces by sbench."""
    df = read_csv(csv)
    df = strip(df, invert=common)
    if pivot:
        index, column = pivot
        df = df.pivot_table(index=index, columns=column, aggfunc=aggregation)
    if auto_group:
        nunique = df.apply(pd.Series.nunique).sort_values()
        groups = [g for g in nunique.index if df.dtypes[g] != float]
        df = df.groupby(groups).agg(aggregation)
    if group:
        df = df.groupby(list(group)).agg(aggregation)
    if select:
        df = df[select[0] if len(select) == 1 else list(select)]
    if sort:
        df = df.sort_values()

    click.echo(df.to_string())
    if output:
        df.to_csv(output)


@main.command()
@click.argument('csv', type=click.Path(exists=True))
@click.option('--best-only/--all', '-b', default=False)
@click.option('--select', multiple=True, default=['time', 'bandwidth'])
@click.option('--separate-by')
def summary(csv, best_only, select, separate_by):
    """Print same data summary as the sbench executable does by default."""
    df = read_csv(csv)
    nunique = df.apply(pd.Series.nunique)
    if len(df.index) > 1:
        df.drop(nunique[nunique <= 1].index, axis=1, inplace=True)

    def show(df):
        groups = list(set(df.columns) - set(select))
        if groups:
            medians = df.groupby(groups).median()
            if best_only:
                best = medians['time'].idxmin()
                click.echo(medians.loc[[best]])
            else:
                click.echo(medians.sort_values(by='time').to_string())
        else:
            click.echo(df.median().to_string())

    if separate_by:
        values = df[separate_by].unique()
        for value in values:
            click.echo(f'{separate_by} = {value}:')
            show(df.loc[df[separate_by] == value])
    else:
        show(df)


@main.command()
@click.argument('csv', type=click.Path(exists=True))
@click.argument('x')
@click.argument('y')
@click.option('--labels', '-l', help='CSV column to use as labels.')
@click.option('--aggregation',
              default='median',
              help='Aggregation function to use in pivotting.')
@click.option('--uniform/--non-uniform',
              '-u',
              help='Equidistant placement of x-axis ticks.')
@click.option('--ylim', type=float, nargs=2, help='Y-axis limits.')
@click.option('--title', '-t', help='Plot title.')
@click.option('--group-by',
              '-g',
              multiple=True,
              help='CSV column to use for grouping.')
@click.option('--reference',
              '-r',
              multiple=True,
              help='Reference value in the form `label=value`.')
@click.option('--relative-to',
              type=float,
              help='Plot percentage relative to given value.')
@click.option('--ascii/--no-ascii',
              '-a',
              help='Use ASCII plotting (requires drawilleplot package).')
@click.option('--label-regex',
              help='Search and replace a pattern in the final labels, '
              'input as /pattern/repl/ in Python regex syntax')
@click.option('--output', '-o', type=click.Path(), help='Output file.')
def plot(csv, labels, x, y, aggregation, uniform, ylim, title, group_by,
         reference, relative_to, ascii, label_regex, output):
    """Plot output of sbench.

    X is the data column name for the values used for the x-axis in the plot, Y
    is the column name for the y-axis.
    """
    df = read_csv(csv)
    common = strip(df, invert=True)
    df = strip(df)

    from matplotlib import pyplot as plt, ticker
    import cycler
    if output:
        plt.switch_backend('Agg')
    elif ascii:
        plt.switch_backend('module://drawilleplot')
    plt.style.use('ggplot')

    plt.rcParams['axes.prop_cycle'] = (cycler.cycler(marker=['o', '^', 's']) *
                                       plt.rcParams['axes.prop_cycle'])

    def pivot_and_plot(df, prefix=None):
        df = df.pivot_table(index=x, columns=labels, aggfunc=aggregation)[y]

        xticks = np.arange(len(df.index)) if uniform else df.index.to_numpy()
        if isinstance(df, pd.DataFrame):
            items = df.items()
        else:
            assert isinstance(df, pd.Series)
            items = [(df.name, df)]

        for label, series in items:
            if prefix:
                label = prefix + label
            if label_regex:
                splitter = label_regex[0]
                if label_regex[-1] != splitter:
                    raise ValueError(
                        'expected input in the form /pattern/repl/')
                pattern, repl = label_regex[1:-1].split(splitter, 1)
                label = re.sub(pattern, repl, label)

            values = series.values
            if relative_to:
                values = values / relative_to
            plt.plot(xticks, values, label=label)

        if uniform:
            plt.xticks(xticks, df.index)

    if group_by:
        group_values = [df[g].unique() for g in group_by]
        for group in itertools.product(*group_values):
            mask = functools.reduce(operator.and_,
                                    (df[k] == v
                                     for k, v in zip(group_by, group)))
            pivot_and_plot(df[mask], ' '.join(group) + ': ')
    else:
        pivot_and_plot(df)

    for i, ref in enumerate(reference):
        label, value = ref.split('=', 1)
        value = float(value)
        dashes = (5 * (i + 1) + 3 * i, 3)
        plt.axhline(value, color='k', ls=(0, dashes), label=label)

    plt.legend()
    plt.xlabel(x)
    plt.ylabel(y)
    if ylim:
        plt.ylim(ylim)
    if relative_to:
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    if not title:
        title = os.path.split(csv)[-1]
    plt.title(title)
    subtitle = ', '.join(f'{k}: {v}' for k, v in common.items())
    plt.text(0.5,
             0.005,
             '\n'.join(textwrap.wrap(subtitle, 130)),
             fontsize=6,
             transform=plt.gcf().transFigure,
             ha='center',
             va='bottom')
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()

    if output:
        plt.savefig(output)
    else:
        plt.show()


@main.command()
@click.argument('csv', type=click.Path(exists=True), nargs=-1)
@click.argument('output', type=click.File(mode='w'))
def merge(csv, output):
    """Merge multiple CSV files produces by sbench."""
    dfs = [read_csv(c) for c in csv]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(output)


if __name__ == '__main__':
    main()
