import copy
import itertools
import operator
import re
import sys
import types

import click
import pandas as pd

from . import benchmark
from .tools import common


class ArgRange:
    def __init__(self, name, values):
        self.name = name
        self.values = values


class MaybeRange(click.ParamType):
    name = 'range or value'

    def __init__(self, base_type):
        self.base_type = base_type

    def _parse_comma_range(self, value, param, ctx):
        return [
            self.base_type.convert(v, param, ctx) for v in value.split(',')
        ]

    def _parse_range(self, value, param, ctx):
        try:
            value, step = value.split(':')
        except ValueError:
            step = '1'

        try:
            start, stop = value.split('-')
        except ValueError:
            self.fail(
                f'expected range of {self.base_type.name},'
                f' got {value}', param, ctx)

        start = self.base_type.convert(start, param, ctx)
        stop = self.base_type.convert(stop, param, ctx)

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

    def convert(self, value, param, ctx):
        if isinstance(value, str) and len(
                value) >= 2 and value[0] == '[' and value[-1] == ']':
            value = value[1:-1]
            try:
                name, value = value.split('=')
            except ValueError:
                name = None

            if value:
                if ',' in value:
                    values = self._parse_comma_range(value, param, ctx)
                else:
                    values = self._parse_range(value, param, ctx)
            else:
                values = None

            return ArgRange(name, values)

        return self.base_type.convert(value, param, ctx)


_RANGE_TYPES = {
    int: MaybeRange(click.INT),
    float: MaybeRange(click.FLOAT),
    str: MaybeRange(click.STRING)
}


def _unpack_ranges(kwargs):
    ranges = dict()

    def find_named_ranges(value):
        if isinstance(value, tuple):
            for v in value:
                find_named_ranges(v)
        if isinstance(value, ArgRange):
            if value.name:
                if value.name in ranges:
                    if ranges[value.name] is None:
                        ranges[value.name] = value.values
                    elif value.values is not None and ranges[
                            value.name] != value.values:
                        raise ValueError(f'multiple definitions for '
                                         'argument range "{value.name}"')
                else:
                    ranges[value.name] = value.values
            else:
                ranges[id(value)] = value.values

    for value in kwargs.values():
        find_named_ranges(value)

    for name, values in ranges.items():
        if values is None:
            raise ValueError(
                f'found named range "{name}" with undefined value')

    unpacked_kwargs = []
    for values in itertools.product(*ranges.values()):
        value_map = {k: v for k, v in zip(ranges.keys(), values)}

        def unpack(value):
            if isinstance(value, tuple):
                return tuple(unpack(v) for v in value)
            if isinstance(value, ArgRange):
                if value.name:
                    return value_map[value.name]
                return value_map[id(value)]
            return value

        unpacked_kwargs.append({k: unpack(v) for k, v in kwargs.items()})
    return unpacked_kwargs


def _non_unique_args(unpacked_kwargs):
    keys = unpacked_kwargs[0].keys()
    return {k for k in keys if len(set(v[k] for v in unpacked_kwargs)) > 1}


def _cli_command(bmark):
    command_path = bmark.__module__.split('.')[2:]
    command_name = re.sub('(?!^)([A-Z1-9]+)', r'-\1', bmark.__name__).lower()
    return command_path + [command_name]


def _cli_func(bmark):
    @click.pass_context
    def run_bmark(ctx, **kwargs):
        unpacked_kwargs = _unpack_ranges(kwargs)
        non_unique = _non_unique_args(unpacked_kwargs)
        results = []
        try:
            with common.report_progress(
                    len(unpacked_kwargs) * ctx.obj.executions) as progress:
                for kws in unpacked_kwargs:
                    try:
                        bmark_instance = bmark(**kws)
                    except benchmark.ParameterError as error:
                        if ctx.obj.skip_invalid_parameters:
                            for _ in range(ctx.obj.executions):
                                progress()
                            continue
                        click.echo()
                        click.echo(*error.args)
                        sys.exit(1)

                    non_unique_kws = {
                        k.replace('_', '-'): v
                        for k, v in kws.items() if k in non_unique
                    }

                    for _ in range(ctx.obj.executions):
                        try:
                            result = bmark_instance.run()
                        except benchmark.ExecutionError as error:
                            if ctx.obj.skip_execution_failures:
                                progress()
                                continue
                            click.echo()
                            click.echo(*error.args)
                            sys.exit(1)
                        result.update(non_unique_kws)
                        results.append(result)
                        progress()
        except KeyboardInterrupt:
            pass
        if not results:
            click.echo('no data collected')
            sys.exit(1)
        table = pd.DataFrame(results)
        if ctx.obj.report == 'full':
            click.echo(table.to_string())
        else:
            if non_unique:
                groups = [k.replace('_', '-') for k in non_unique]
                medians = table.groupby(groups).median()
                if ctx.obj.report == 'best-median':
                    best = medians['time'].idxmin()
                    click.echo(medians.loc[[best]])
                else:
                    click.echo(medians.sort_values(by='time').to_string())
            else:
                click.echo(table.median().to_string())
        if ctx.obj.output:
            common.write_csv(table, ctx.obj.output)

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
@click.option('--executions', '-e', type=int, default=3)
@click.option('--report',
              '-r',
              default='best-median',
              type=click.Choice(['best-median', 'all-medians', 'full']))
@click.option('--skip-invalid-parameters', '-s', is_flag=True)
@click.option('--skip-execution-failures', '-q', is_flag=True)
@click.option('--output', '-o', type=click.Path())
@click.pass_context
def _cli(ctx, executions, report, skip_invalid_parameters,
         skip_execution_failures, output):
    ctx.obj.executions = executions
    ctx.obj.report = report
    ctx.obj.skip_invalid_parameters = skip_invalid_parameters
    ctx.obj.skip_execution_failures = skip_execution_failures
    ctx.obj.output = output


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
