import contextlib
import copy
import itertools
import operator
import re
import sys
import types

import click
import pandas as pd

from . import benchmark


class MaybeRange(click.ParamType):
    name = 'range or value'

    def __init__(self, base_type):
        self.base_type = base_type

    def convert(self, value, param, ctx):
        if isinstance(value, str) and value[0] == '[' and value[-1] == ']':
            range_value = value[1:-1]
            if ',' in range_value:
                return [
                    self.base_type.convert(v, param, ctx)
                    for v in range_value.split(',')
                ]

            try:
                start, stop, *step = range_value.split(':')
            except ValueError:
                self.fail(
                    f'expected range of {self.base_type.name},'
                    ' got {value}', param, ctx)

            start = self.base_type.convert(start, param, ctx)
            stop = self.base_type.convert(stop, param, ctx)
            if step:
                if len(step) == 1:
                    step = step[0]
                else:
                    self.fail(
                        f'expected range of {self.base_type.name}, got '
                        ' invalid step in {value}', param, ctx)
            else:
                step = '1'

            range_op = operator.add
            valid_ops = {
                '+': operator.add,
                '-': operator.sub,
                '*': operator.mul,
                '/': operator.truediv
            }
            if step[0] in valid_ops:
                range_op = valid_ops[step[0]]
                step = step[1:]
            step = self.base_type.convert(step, param, ctx)
            result = []
            current = start
            while start <= current <= stop:
                result.append(current)
                current = range_op(current, step)
            return result

        return self.base_type.convert(value, param, ctx)


_RANGE_TYPES = {
    int: MaybeRange(click.INT),
    float: MaybeRange(click.FLOAT),
    str: MaybeRange(click.STRING)
}


def _unpack_ranges(**kwargs):
    def _unpack(arg):
        if isinstance(arg, list):
            return arg
        elif isinstance(arg, tuple):
            return list(itertools.product(*(_unpack(a) for a in arg)))
        else:
            return [arg]

    unpacked = [_unpack(arg) for arg in kwargs.values()]
    non_unique = tuple(k for k, v in zip(kwargs.keys(), unpacked)
                       if len(v) > 1)
    return [{k: v
             for k, v in zip(kwargs.keys(), args)}
            for args in itertools.product(*unpacked)], non_unique


def _cli_command(bmark):
    command_path = bmark.__module__.split('.')[2:]
    command_name = re.sub('(?!^)([A-Z1-9]+)', r'-\1', bmark.__name__).lower()
    return command_path + [command_name]


@contextlib.contextmanager
def _report_progress(total):
    current = 0

    def report():
        nonlocal current
        percent = min(100 * current // total, 100)
        click.echo('#' * percent + '-' * (100 - percent) + f' {percent:3}%\r',
                   nl=False)
        current += 1

    report()
    yield report
    click.echo(' ' * 110 + '\r')


def _cli_func(bmark):
    @click.pass_context
    def run_bmark(ctx, **kwargs):
        unpacked_kwargs, non_unique = _unpack_ranges(**kwargs)
        tables = []
        try:
            with _report_progress(len(unpacked_kwargs) *
                                  ctx.obj.executions) as progress:
                for kws in unpacked_kwargs:
                    try:
                        bmark_instance = bmark(**kws)
                    except benchmark.ParameterError as error:
                        click.echo()
                        click.echo(*error.args)
                        sys.exit(1)

                    results = []
                    for _ in range(ctx.obj.executions):
                        results.append(bmark_instance.run())
                        progress()

                    table = pd.DataFrame(results)

                    for arg in non_unique:
                        table[arg] = [kws[arg]] * len(table.index)

                    tables.append(table)
        except KeyboardInterrupt:
            pass
        full_table = pd.concat(tables, ignore_index=True)
        if ctx.obj.report == 'full':
            click.echo(full_table.to_string())
        else:
            if non_unique:
                medians = full_table.groupby(list(non_unique)).median()
                if ctx.obj.report == 'best-median':
                    best = medians['time'].idxmin()
                    click.echo(medians.loc[[best]])
                else:
                    click.echo(medians.to_string())
            else:
                click.echo(full_table.median().to_string())

    func = run_bmark
    for name, param in bmark.parameters.items():
        name = '--' + name.replace('_', '-')
        if param.dtype is bool:
            option = click.option(name + '/' + name.replace('--', '--no-'),
                                  default=param.default,
                                  help=param.description)
        else:
            option = click.option(name,
                                  type=_RANGE_TYPES[param.dtype],
                                  nargs=param.nargs,
                                  help=param.description,
                                  required=param.default is None,
                                  default=param.default)
        func = option(func)
    return func


@click.group()
@click.option('--executions', '-e', type=int, default=1)
@click.option('--report',
              '-r',
              default='best-median',
              type=click.Choice(['best-median', 'all-medians', 'full']))
@click.pass_context
def _cli(ctx, executions, report):
    ctx.obj.executions = executions
    ctx.obj.report = report


def _build(commands):
    hierarchy = dict()

    for command, func in commands:
        current = hierarchy
        for subcommand in command[:-1]:
            current = current.setdefault(subcommand, dict())
        current[command[-1]] = func

    @click.pass_context
    def empty(_):
        pass

    def build_click_hierarchy(group, subcommands):
        for subcommand, subcommand_subcommands in subcommands.items():
            if isinstance(subcommand_subcommands, dict):
                build_click_hierarchy(
                    group.group(name=subcommand.replace('_', '-'))(empty),
                    subcommand_subcommands)
            else:
                group.command(name=subcommand)(subcommand_subcommands)

    main_group = copy.copy(_cli)
    build_click_hierarchy(main_group, hierarchy)
    return main_group


def main():
    commands = [(_cli_command(bmark), _cli_func(bmark))
                for bmark in benchmark.REGISTRY]
    main_group = _build(commands)

    def func(*args, **kwargs):
        return main_group(*args, **kwargs, obj=types.SimpleNamespace())

    return func
