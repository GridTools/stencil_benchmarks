# Stencil Benchmarks
#
# Copyright (c) 2017-2021, ETH Zurich and MeteoSwiss
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

import os
import re
import textwrap

import click
import numpy as np
import pandas as pd

pd.set_option('display.max_colwidth', None)


@click.group()
def main():
    pass


def read_csv(csv):
    return pd.read_csv(csv,
                       index_col=0,
                       na_values=['nan', 'NaN'],
                       keep_default_na=False)


def split(df):
    nunique = df.apply(pd.Series.nunique)
    varying = df.drop(nunique[nunique == 1].index, axis='columns')
    common = df.drop(nunique[nunique > 1].index, axis='columns')
    common = common.iloc[0].sort_index()
    return varying, common


def auto_groups(df):
    nunique = df.apply(pd.Series.nunique).sort_values()
    return [g for g in nunique.index if df.dtypes[g] != float]


def arrange(df, query, groups, aggregation, unstack, select, sort):
    if query:
        df = df.query(query)
    df, common = split(df)
    if groups == 'auto':
        groups = auto_groups(df)
    if groups:
        df = df.groupby(list(groups)).agg(aggregation)
    if unstack:
        df = df.unstack()
    if select:
        df = df[select[0] if len(select) == 1 else list(select)]
    if sort:
        df = df.sort_values()
    return df, common


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
@click.option('--unstack/--dont-unstack',
              '-u',
              default=False,
              help='‘Unstack’ the last group, useful for pivot-tables.')
@click.option('--select',
              '-s',
              multiple=True,
              help='Only print values from selected columns.')
@click.option('--filter', '-f', help='Filter values based on a condition.')
@click.option('--aggregation',
              default='median',
              help='Aggregation function used for pivotting.')
@click.option('--sort/--no-sort', default=False, help='Sort values.')
@click.option('--output', '-o', type=click.File(mode='w'), help='Output file.')
def print_csv(csv, common, auto_group, group, unstack, select, filter,
              aggregation, sort, output):
    """Print reports of CSV files produces by sbench."""
    df = read_csv(csv)
    groups = 'auto' if auto_group else group
    df, dfc = arrange(df, filter, groups, aggregation, unstack, select, sort)
    if common:
        df = dfc

    click.echo(df.to_string())
    if output:
        df.to_csv(output)


@main.command()
@click.argument('csv', type=click.Path(exists=True))
@click.option('--uniform/--non-uniform',
              help='Equidistant placement of x-axis ticks.')
@click.option('--ylim', type=float, nargs=2, help='Y-axis limits.')
@click.option('--title', '-t', help='Plot title.')
@click.option('--auto-group/--no-auto-group',
              '-a',
              default=False,
              help='Automatically group values.')
@click.option('--group',
              '-g',
              multiple=True,
              help='CSV column to use for grouping.')
@click.option('--select', '-s', required=True, help='Values to plot.')
@click.option('--filter', '-f', help='Filter values based on a condition.')
@click.option('--aggregation',
              default='median',
              help='Aggregation function to use.')
@click.option('--reference',
              '-r',
              multiple=True,
              help='Reference value in the form `label=value`.')
@click.option('--relative-to',
              type=float,
              help='Plot percentage relative to given value.')
@click.option('--ascii/--no-ascii',
              help='Use ASCII plotting (requires drawilleplot package).')
@click.option('--label-regex',
              multiple=True,
              help='Search and replace a pattern in the final labels, '
              'input as /pattern/repl/ in Python regex syntax.')
@click.option('--output', '-o', type=click.Path(), help='Output file.')
@click.option('--dpi',
              default=300,
              type=int,
              help='Output DPI (dots per inch).')
def plot(csv, uniform, ylim, title, auto_group, group, select, filter,
         aggregation, reference, relative_to, ascii, label_regex, output, dpi):
    """Plot output of sbench.

    X is the data column name for the values used for the x-axis in the plot, Y
    is the column name for the y-axis.
    """
    import cycler
    from matplotlib import pyplot as plt
    from matplotlib import ticker
    if output:
        plt.switch_backend('Agg')
    elif ascii:
        plt.switch_backend('module://drawilleplot')
    plt.style.use('ggplot')

    plt.rcParams['axes.prop_cycle'] = (
        cycler.cycler(linestyle=['-', '--', '-.', ':']) *
        cycler.cycler(marker=['o', '^', 's']) *
        plt.rcParams['axes.prop_cycle'])

    df = read_csv(csv)
    groups = 'auto' if auto_group else group
    df, common = arrange(df, filter, groups, aggregation, True, [select],
                         False)

    regexes = []
    for regex in label_regex:
        splitter = regex[0]
        if regex[-1] != splitter:
            raise ValueError('expected input in the form /pattern/repl/')
        pattern, repl = regex[1:-1].split(splitter, 1)
        regexes.append((re.compile(pattern), repl))

    for index, row in df.iterrows():
        x = np.arange(len(row.index)) if uniform else row.index
        y = row.values / relative_to if relative_to else row.values
        if isinstance(index, tuple):
            label = ', '.join(f'{name}={value}'
                              for name, value in zip(df.index.names, index))
        else:
            label = str(index)
        for regex, repl in regexes:
            label = regex.sub(repl, label)
        plt.plot(x, y, label=label)
    if uniform:
        plt.xticks(x, row.index, rotation=45)

    for i, ref in enumerate(reference):
        label, value = ref.split('=', 1)
        value = float(value)
        dashes = (5 * (i + 1) + 3 * i, 3)
        plt.axhline(value, color='k', ls=(0, dashes), label=label)

    plt.legend()
    plt.xlabel(df.columns.name)
    plt.ylabel(select)
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
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=dpi)
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
