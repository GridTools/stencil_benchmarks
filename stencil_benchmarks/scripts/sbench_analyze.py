#!/usr/bin/env python

import functools
import itertools
import operator
import os
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
@click.option('--common/--non-common', '-c', default=False)
@click.option('--auto-group/--no-auto-group', '-a', default=False)
@click.option('--group', '-g', multiple=True)
@click.option('--select', '-s', multiple=True)
@click.option('--pivot', '-p', nargs=2)
@click.option('--aggregation', default='median')
@click.option('--output', '-o', type=click.Path())
def print_csv(csv, common, auto_group, group, select, pivot, aggregation,
              output):
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
        df = df.groupby(list(groups)).agg(aggregation)
    if select:
        df = df[select[0] if len(select) == 1 else list(select)]

    click.echo(df.to_string())
    if output:
        df.to_csv(output)


@main.command()
@click.argument('csv', type=click.Path(exists=True))
@click.argument('by')
@click.argument('x')
@click.argument('y')
@click.option('--aggregation', default='median')
@click.option('--uniform/--non-uniform', '-u')
@click.option('--ylim', type=float, nargs=2)
@click.option('--title', '-t')
@click.option('--group-by', '-g', multiple=True)
@click.option('--output', '-o', type=click.Path())
def plot(csv, by, x, y, aggregation, uniform, ylim, title, group_by, output):
    df = read_csv(csv)
    common = strip(df, invert=True)
    df = strip(df)

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.style.use('ggplot')

    def pivot_and_plot(df, prefix=None):
        df = df.pivot_table(index=x, columns=by, aggfunc=aggregation)[y]

        xticks = np.arange(len(df.index)) if uniform else df.index
        for label, values in df.items():
            if prefix:
                label = prefix + label
            plt.plot(xticks, values.values, marker='o', label=label)
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

    plt.legend()
    plt.xlabel(x)
    plt.ylabel(y)
    if ylim:
        plt.ylim(ylim)
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
@click.argument('output', type=click.Path())
def merge(csv, output):
    dfs = [read_csv(c) for c in csv]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(output)


if __name__ == '__main__':
    main()
