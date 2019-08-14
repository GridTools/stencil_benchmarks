import copy
import re
import sys
import types

import click
import pandas as pd

from . import benchmark


def _cli_command(bmark):
    command_path = bmark.__module__.split('.')[2:]
    command_name = re.sub('(?!^)([A-Z1-9]+)', r'-\1', bmark.__name__).lower()
    return command_path + [command_name]


def _cli_func(bmark):
    @click.pass_context
    def run_bmark(ctx, **kwargs):
        try:
            bmark_instance = bmark(**kwargs)
        except benchmark.ParameterError as error:
            print(*error.args)
            sys.exit(1)
        results = [bmark_instance.run() for _ in range(ctx.obj.executions)]

        table = pd.DataFrame(results)
        if ctx.obj.metric:
            table = table[list(ctx.obj.metric)]
        click.echo(table.agg(ctx.obj.aggregation).T)

    func = run_bmark
    for name, param in bmark.parameters.items():
        name = '--' + name.replace('_', '-')
        if param.dtype is bool:
            option = click.option(name + '/' + name.replace('--', '--no-'),
                                  default=param.default,
                                  help=param.description)
        else:
            option = click.option(name,
                                  type=param.dtype,
                                  nargs=param.nargs,
                                  help=param.description,
                                  required=param.default is None,
                                  default=param.default)
        func = option(func)
    return func


@click.group()
@click.option('--executions', '-e', type=int, default=1)
@click.option('--aggregation',
              '-a',
              multiple=True,
              default=['median', 'min', 'max'])
@click.option('--metric', '-m', multiple=True)
@click.pass_context
def _cli(ctx, executions, aggregation, metric):
    ctx.obj.executions = executions
    ctx.obj.aggregation = aggregation
    ctx.obj.metric = metric


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
