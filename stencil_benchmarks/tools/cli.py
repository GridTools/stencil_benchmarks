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
"""Helper functions for click CLI applications."""

import itertools
import operator
import re
import sys
import types
from typing import (Any, Dict, Iterable, Iterator, List, NamedTuple, Optional,
                    Tuple)

import click

from .. import __version__


def colorize(string: str,
             rgb: Optional[Tuple[float, float, float]] = None,
             hsv: Optional[Tuple[float, float, float]] = None) -> str:
    """Add truecolor terminal color code to `string`.

    Either RGB or HSV values must be given.

    Parameters
    ----------
    string : str
        String to colorize.
    rgb : None or tuple of float
        3-tuple of RGB values in range [0, 1].
    hsv : None or tuple of float
        3-tuple of HSV values in range [0, 1].

    Returns
    -------
    str
        Input string surrounded by truecolor terminal color code.

    Examples
    --------
    >>> colorize('foo', rgb=(1, 0, 0)) # doctest: +ELLIPSIS
    '...foo...'
    >>> colorize('foo', hsv=(1, 0, 0)) # doctest: +ELLIPSIS
    '...foo...'
    """
    if hsv:
        h, s, v = hsv

        def f(n):
            k = (n + h * 6) % 6
            return v - v * s * max(min(k, 4 - k, 1), 0)

        rgb = f(5), f(3), f(1)

    if not rgb:
        raise ValueError('missing color specification')

    r, g, b = (int(255 * x) for x in rgb)
    return f'\x1b[38;2;{r};{g};{b}m{string}\x1b[0m'


class ProgressBar:
    """Simple CLI progress bar.

    Examples
    --------
    >>> p = ProgressBar()
    """
    def __init__(self):
        self._progress = []

    def report(self, iterable: Iterable[Any]) -> Iterator[Any]:
        """Report progress when looping over `iterable`.

        Reports progress on an iterable. Nested application possible.

        Parameters
        ----------
        iterable : iterable
            Iterable to wrap.

        Returns
        -------
        list
            `iterable` converted into a list.

        Examples
        --------
        >>> import time
        >>> with ProgressBar() as p:
        ...     for i in p.report(range(5)):
        ...         for j in p.report(range(2)):
        ...             time.sleep(0.1)
        \r...
        """
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
    def progress(self) -> float:
        """Get the current progress in percent.

        Value is only meaningful inside a report() loop.

        Returns
        -------
        float
            Current progress in percent.

        Example
        -------
        >>> p = ProgressBar()
        >>> p.progress
        0.0
        """
        percent = 0.0
        width = 100.0
        for sub_progress in self._progress:
            percent += sub_progress.current / sub_progress.max * width
            width /= sub_progress.max
        return percent

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point.

        Clears progress bar in case of succes,
        otherwise prints a newline.
        """
        if exc_type is None:
            click.echo('\r' + ' ' * 110 + '\r', nl=False)
        else:
            click.echo()
        return False

    def _print(self) -> None:
        percent = round(self.progress)
        bar = colorize('█' * percent + '-' * (100 - percent),
                       hsv=(percent / 300, 1, 1))
        click.echo('\r|' + bar + f'| {percent:3}%', nl=False)
        sys.stdout.flush()


class ArgRange(NamedTuple):
    """Argument range, i.e. range name and associated values.

    Properties
    ----------
    name : str
        Identifier of the range.
    values : tuple or None
        Tuple of all values in the range or None if unknown.
    """

    name: str
    values: Optional[tuple]

    def unique_values(self, *others) -> tuple:
        """Collect unique values definition from multiple occurences of a range.

        Examples
        --------
        >>> x = ArgRange('foo', (1, 2, 3))
        >>> y = ArgRange('foo', (1, 2, 3))
        >>> z = ArgRange('foo', None)
        >>> x.unique_values(y, z)
        (1, 2, 3)
        >>> x.unique_values()
        (1, 2, 3)
        >>> z.unique_values()
        Traceback (most recent call last):
            ...
        ValueError: missing value definition for argument range "foo"
        """
        assert all(self.name == other.name for other in others)
        all_ranges = (self, ) + others
        values = set(r.values for r in all_ranges if r.values)
        if not values:
            raise ValueError(
                f'missing value definition for argument range "{self.name}"')
        if len(values) > 1:
            raise ValueError(
                f'multiple definitions for argument range "{self.name}"')
        return next(iter(values))


class _MaybeRange(click.ParamType):
    """Click ParamType for parameters that accept ranges."""

    name = 'maybe range'

    def __init__(self, base_type):
        self.base_type = base_type
        self._anonymous_range_count = 0

    def _parse_comma_range(self, value, param, ctx):
        return tuple(
            self.base_type.convert(v, param, ctx) for v in value.split(','))

    def _parse_range(self, value, param, ctx):
        match = re.match(
            r'^(?P<start>[+-]?\d+)-(?P<stop>[+-]?\d+)'
            r'(:(?P<op>[*/+-])(?P<step>[+-]?\d+))?$', value)

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
                values = None

            return ArgRange(name, values)

        return self.base_type.convert(value, param, ctx)

    def get_metavar(self, param):
        base_var = self.base_type.get_metavar(param)
        if not base_var:
            base_var = self.base_type.name
        return f'{base_var} [or {base_var} range]'


_RANGE_TYPES = {
    int: _MaybeRange(click.INT),
    float: _MaybeRange(click.FLOAT),
    str: _MaybeRange(click.STRING)
}


def range_type(dtype: type) -> _MaybeRange:
    """Get click range type.

    The range type allows parsing click options and arguments in range form,
    that allow to easily specify a whole range of values for a single option.

    To be used with `unpack_ranges`.

    Parameters
    ----------
    dtype : int or float or str
        Range value type.

    Returns
    -------
    range object
        A range param object, that can be used
        in click.argument() and click.option().

    Examples
    --------
    Create a click.command that accepts range args:
    >>> @click.command()
    ... @click.argument('arg', type=range_type(int))
    ... @click.option('--option', type=range_type(int))
    ... def foo(arg, option):
    ...     return arg, option

    Calling the function with normal args behaves as normal:
    >>> foo(args=['1', '--option', '3'], standalone_mode=False)
    (1, 3)

    Calling the function with range arguments gives ArgRange objects:
    >>> foo(args=['[x=1-3]', '--option', '[3,4]'], standalone_mode=False)
    (ArgRange(name='x', values=(1, 2, 3)), ArgRange(name='...', values=(3, 4)))
    """
    if isinstance(dtype, click.ParamType):
        return _MaybeRange(dtype)
    return _RANGE_TYPES[dtype]


def _extract_ranges(values: Iterable[Any]) -> Iterator[ArgRange]:
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


def unpack_ranges(**kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Unpack range arguments in `kwargs`.

    Generates a list of kwargs of the cartesian product of all range
    values inside kwargs.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to unpack, possibly with `ArgRange` values.

    Returns
    -------
    list of dict
        List of kwargs, with all `ArgRanges` unpacked.

    Examples
    --------
    >>> unpack_ranges(a=3, b=5)
    [{'a': 3, 'b': 5}]
    >>> unpack_ranges(a=3, b=ArgRange(name='r0', values=(1, 2, 3)))
    [{'a': 3, 'b': 1}, {'a': 3, 'b': 2}, {'a': 3, 'b': 3}]
    >>> unpack_ranges(a=ArgRange(name='r0', values=(1, 2)),
    ...               b=ArgRange(name='r1', values=(3, 4)))
    [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    >>> unpack_ranges(a=ArgRange(name='r0', values=(1, 2)),
    ...               b=ArgRange(name='r0', values=None))
    [{'a': 1, 'b': 1}, {'a': 2, 'b': 2}]
    """
    ranges = _extract_ranges(kwargs.values())
    range_name = operator.attrgetter('name')
    grouped_ranges = itertools.groupby(sorted(ranges, key=range_name),
                                       key=range_name)

    range_map = {k: ArgRange.unique_values(*v) for k, v in grouped_ranges}

    return [{k: _map_ranges(v, value_dict)
             for k, v in kwargs.items()}
            for value_dict in _values_from_range_map(range_map)]


def range_args(**kwargs: Dict[str, Any]) -> Iterator[str]:
    """Iterate over all range arguments in `kwargs`.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to iterate on.

    Yields
    ------
    generator of str
        Names of the range arguments in `kwargs`.

    Example
    -------
    >>> [x for x in range_args(a=1, b=2)]
    []
    >>> [x for x in range_args(a=1, b=ArgRange(name='r0', values=(1, 2)))]
    ['b']
    """
    for arg, value in kwargs.items():
        if isinstance(value, ArgRange) or isinstance(value, tuple) and any(
                isinstance(v, ArgRange) for v in value):
            yield arg


def pretty_parameters(bmark,
                      include_version: bool = True,
                      include_name: bool = True) -> Dict[str, Any]:
    """Get pretty formatted dict of benchmark parameters.

    Parameters
    ----------
    bmark : stencil_benchmarks.benchmark.Benchmark
        Benchmark instance to retrieve parameters from.
    include_version : bool
        Include stencil_benchmarks version information.

    Returns
    -------
    dict
        Dict of all parameter names and values of `bmark`,
        with prettyfied names and unpacked tuple values.

    Examples
    --------
    >>> from stencil_benchmarks.benchmark import (
    ...     Benchmark, Parameter)
    >>> class BMark(Benchmark):
    ...     foo_bar = Parameter('foo param', dtype=int, nargs=1)
    ...     baz = Parameter('bar param', default=(1, 2))
    ...     def run(self):
    ...         pass
    >>> bmark = BMark(foo_bar=42)
    >>> pretty_parameters(bmark, include_version=False)
    {'foo-bar': 42, 'baz-0': 1, 'baz-1': 2}
    """
    parameters = dict()
    for name, value in bmark.parameters.items():
        name = name.replace('_', '-')
        if isinstance(value, tuple):
            for i, v in enumerate(value):
                parameters[f'{name}-{i}'] = v
        else:
            parameters[name] = value
    if include_version:
        parameters['sbench-version'] = __version__
    if include_name:
        parameters['benchmark-name'] = (type(bmark).__module__ + '.' +
                                        type(bmark).__name__)
    return parameters
