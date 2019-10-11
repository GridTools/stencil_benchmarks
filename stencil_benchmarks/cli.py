import copy
import re
import sys
import types

import click
import pandas as pd

from . import benchmark
from .tools import cli as cli_tools, validation


def _cli_command(bmark):
    command_path = bmark.__module__.split('.')[2:]
    command_name = re.sub('(?!^)([A-Z1-9]+)', r'-\1', bmark.__name__).lower()
    return command_path + [command_name]


def _cli_func(bmark):
    @click.pass_context
    def run_bmark(ctx, **kwargs):
        unpacked_kwargs = list(cli_tools.unpack_ranges(**kwargs))
        result_keys = None
        results = []
        try:
            with cli_tools.ProgressBar() as progress:
                for kws in progress.report(unpacked_kwargs):
                    try:
                        bmark_instance = bmark(**kws)
                    except benchmark.ParameterError as error:
                        if ctx.obj.skip_invalid_parameters:
                            continue
                        click.echo(*error.args)
                        sys.exit(1)

                    for _ in progress.report(range(ctx.obj.executions)):
                        try:
                            result = bmark_instance.run()
                            if result_keys is None:
                                result_keys = set(result.keys())
                        except benchmark.ExecutionError as error:
                            if ctx.obj.skip_execution_failures:
                                continue
                            click.echo(*error.args)
                            sys.exit(1)
                        except validation.ValidationError:
                            click.echo('error: validation failed')
                            sys.exit(2)
                        result.update(
                            cli_tools.pretty_parameters(bmark_instance))
                        results.append(result)
        except KeyboardInterrupt:
            pass
        if not results:
            click.echo('no data collected')
            sys.exit(1)
        table = pd.DataFrame(results)
        if ctx.obj.output:
            table.to_csv(ctx.obj.output)

        nunique = table.apply(pd.Series.nunique)
        if len(table.index) > 1:
            table.drop(nunique[nunique <= 1].index, axis=1, inplace=True)

        if ctx.obj.report == 'full':
            click.echo(table.to_string())
        else:
            group_keys = list(set(table.columns) - result_keys)
            if group_keys:
                medians = table.groupby(group_keys).median()
                if ctx.obj.report == 'best-median':
                    best = medians['time'].idxmin()
                    click.echo(medians.loc[[best]])
                else:
                    click.echo(medians.sort_values(by='time').to_string())
            else:
                click.echo(table.median().to_string())

    func = run_bmark
    for name, param in bmark.parameters.items():
        name = '--' + name.replace('_', '-')
        if param.dtype is bool:
            option = click.option(name + '/' + name.replace('--', '--no-'),
                                  default=param.default,
                                  help=param.description)
        else:
            option = click.option(name,
                                  type=cli_tools.range_type(param.dtype),
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
