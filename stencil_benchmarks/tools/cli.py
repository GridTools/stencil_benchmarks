import contextlib
import itertools
import operator
import sys
import typing

import click


@contextlib.contextmanager
def report_progress(total):
    current = 0

    def report():
        nonlocal current
        percent = min(int(100 * current / total), 100)
        click.echo('\r' + '#' * percent + '-' * (100 - percent) +
                   f' {percent:3}%',
                   nl=False)
        sys.stdout.flush()
        current += 1

    click.echo()
    report()
    yield report
    click.echo('\r' + ' ' * 110 + '\r', nl=False)


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


def range(dtype):
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
