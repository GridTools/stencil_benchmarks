import itertools
import operator
import re
import sys
import types
import typing

import click


class ProgressBar:
    def __init__(self):
        self._progress = []

    def report(self, iterable):
        iterable = list(iterable)
        index = len(self._progress)
        self._progress.append(
            types.SimpleNamespace(current=0, max=len(iterable)))
        self._print()
        try:
            for i in iterable:
                yield i
                self._progress[index].current += 1
                self._print()
        finally:
            assert len(self._progress) == index + 1
            self._progress.pop()

    @property
    def progress(self):
        percent = 0.0
        width = 100.0
        for sub_progress in self._progress:
            percent += sub_progress.current / sub_progress.max * width
            width /= sub_progress.max
        return percent

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            click.echo('\r' + ' ' * 110 + '\r', nl=False)
        else:
            click.echo()
        return False

    def _print(self):
        percent = round(self.progress)
        click.echo('\r' + '#' * percent + '-' * (100 - percent) +
                   f' {percent:3}%',
                   nl=False)
        sys.stdout.flush()


class ArgRange(typing.NamedTuple):
    name: str
    values: tuple

    def unique_values(self, *others):
        all_ranges = (self, ) + others
        values = set(r.values for r in all_ranges if r.values)
        if not values:
            raise ValueError(
                f'missing value definition for argument range "{self.name}')
        if len(values) > 1:
            raise ValueError(
                f'multiple definitions for argument range "{self.name}"')
        return next(iter(values))


class MaybeRange(click.ParamType):
    name = 'range or value'

    def __init__(self, base_type):
        self.base_type = base_type
        self._anonymous_range_count = 0

    def _parse_comma_range(self, value, param, ctx):
        return tuple(
            self.base_type.convert(v, param, ctx) for v in value.split(','))

    def _parse_range(self, value, param, ctx):
        match = re.match(r'^(?P<start>[+-]?\d+)-(?P<stop>[+-]?\d+)(:(?P<op>[*/+-])(?P<step>[+-]?\d+))?$', value)

        if not match:
            self.fail(f'could not parse range "{value}"')

        start = self.base_type.convert(match.group('start'), param, ctx)
        stop = self.base_type.convert(match.group('stop'), param, ctx)

        range_op = operator.add
        valid_ops = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv
        }
        if match.group('op'):
            range_op = valid_ops[match.group('op')]
        if match.group('step'):
            step = self.base_type.convert(match.group('step'), param, ctx)
        else:
            step = 1
        result = []
        current = start
        while start <= current <= stop:
            result.append(current)
            current = range_op(current, step)
        return tuple(result)

    def convert(self, value, param, ctx):
        if isinstance(value, str) and len(
                value) >= 2 and value[0] == '[' and value[-1] == ']':
            value = value[1:-1]
            try:
                name, value = value.split('=')
            except ValueError:
                name = f'{id(self)}_{self._anonymous_range_count}'
                self._anonymous_range_count += 1

            if value:
                if ',' in value:
                    values = self._parse_comma_range(value, param, ctx)
                else:
                    values = self._parse_range(value, param, ctx)
            else:
                values = ()

            return ArgRange(name, values)

        return self.base_type.convert(value, param, ctx)


_RANGE_TYPES = {
    int: MaybeRange(click.INT),
    float: MaybeRange(click.FLOAT),
    str: MaybeRange(click.STRING)
}


def range_type(dtype):
    return _RANGE_TYPES[dtype]


def _extract_ranges(values):
    for value in values:
        if isinstance(value, ArgRange):
            yield value
        elif isinstance(value, tuple):
            yield from _extract_ranges(value)


def _map_ranges(value, value_map):
    if isinstance(value, ArgRange):
        return value_map[value.name]
    if isinstance(value, tuple):
        return tuple(_map_ranges(v, value_map) for v in value)
    return value


def _values_from_range_map(range_map):
    for values in itertools.product(*range_map.values()):
        yield dict(zip(range_map.keys(), values))


def unpack_ranges(**kwargs):
    ranges = _extract_ranges(kwargs.values())
    range_name = operator.attrgetter('name')
    grouped_ranges = itertools.groupby(sorted(ranges, key=range_name),
                                       key=range_name)

    range_map = {k: ArgRange.unique_values(*v) for k, v in grouped_ranges}

    return [{k: _map_ranges(v, value_dict)
             for k, v in kwargs.items()}
            for value_dict in _values_from_range_map(range_map)]


def range_args(**kwargs):
    for arg, value in kwargs.items():
        if isinstance(value, ArgRange) or isinstance(value, tuple) and any(
                isinstance(v, ArgRange) for v in value):
            yield arg


def pretty_parameters(bmark):
    parameters = dict()
    for name, value in bmark.parameters.items():
        name = name.replace('_', '-')
        if isinstance(value, tuple):
            for i, v in enumerate(value):
                parameters[f'{name}-{i}'] = v
        else:
            parameters[name] = value
    return parameters
