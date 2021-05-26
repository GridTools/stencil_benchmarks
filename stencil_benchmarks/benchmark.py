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
import abc
import inspect
from typing import Any, Dict, Iterable, List, Optional, Type, Union

# pylint: disable=arguments-differ,access-member-before-definition

REGISTRY = set()


class ParameterError(ValueError):
    pass


class ExecutionError(RuntimeError):
    pass


class Parameter:
    def __init__(self,
                 description: str,
                 default: Any = None,
                 dtype: Optional[Type] = None,
                 nargs: Optional[int] = None,
                 choices: Optional[Iterable] = None):
        if default is None:
            if dtype is None or nargs is None:
                raise ValueError(
                    'dtype and nargs must be given if default is None')
        else:
            if isinstance(default, (tuple, list)):
                if not default:
                    raise ValueError('can not use empty tuple as default')
                default_dtypes = set(type(d) for d in default)
                if len(default_dtypes) > 1:
                    raise ValueError('different types in default tuple')
                expected_dtype = next(iter(default_dtypes))
                expected_nargs = len(default)
            else:
                expected_dtype = type(default)
                expected_nargs = 1

            if dtype is None:
                dtype = expected_dtype
            elif dtype is not expected_dtype:
                raise ValueError('iconsistent default and dtype values')

            if nargs is None:
                nargs = expected_nargs
            elif nargs != expected_nargs:
                raise ValueError('inconsistent default and nargs values')

        self.description = description
        self.default = default
        self.dtype = dtype
        self.nargs = nargs
        self.choices = tuple(choices) if choices is not None else None

    def validate(self, value):
        if value is None:
            if self.default is None:
                raise ParameterError('value is required')
            value = self.default

        if self.nargs == 1:
            if not isinstance(value, self.dtype):
                raise ParameterError(
                    f'wrong type of argument "{value}", '
                    f' expected one of type "{self.dtype.__name__}"')
        else:
            if not isinstance(value, (tuple, list)):
                raise ParameterError(
                    f'{self.nargs} arguments of type '
                    f'"{self.dtype.__name__}" required, found "{value}"')
            if len(value) != self.nargs:
                raise ParameterError(
                    f'wrong number of arguments in argument "{value}"')
            for val in value:
                if not isinstance(val, self.dtype):
                    raise ParameterError(f'wrong type in argument "{value}"')

        if self.choices is not None and value not in self.choices:
            choices_str = ', '.join(f'"{choice}"' for choice in self.choices)
            raise ParameterError(f'unsupported argument value "{value}", '
                                 f'choices are {choices_str}')
        return value

    def __repr__(self):
        return (f'{type(self).__name__}('
                f'description={self.description!r}, '
                f'dtype={self.dtype!r}, '
                f'nargs={self.nargs!r}, '
                f'default={self.default!r})')

    def __eq__(self, other):
        return (self.description == other.description
                and self.dtype is other.dtype and self.nargs == other.nargs
                and self.default == other.default)


class BenchmarkMeta(abc.ABCMeta):
    def __new__(cls, name, bases, namespace):
        if 'parameters' in namespace:
            raise AttributeError(
                'Benchmark classes must not define an attribute `parameters`')

        parameters = dict()

        for base in bases:
            if hasattr(base, 'parameters'):
                parameters.update(base.parameters)

        transformed_attr_dict = dict()
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, Parameter):
                parameters[attr_name] = attr_value
            else:
                transformed_attr_dict[attr_name] = attr_value

        transformed_attr_dict['parameters'] = parameters
        benchmark = super().__new__(cls, name, bases, transformed_attr_dict)
        if not inspect.isabstract(benchmark):
            REGISTRY.add(benchmark)
        return benchmark


class Benchmark(metaclass=BenchmarkMeta):
    def __init__(self, **kwargs):
        unsupported_args = set(kwargs) - set(self.parameters)
        if unsupported_args:
            raise ValueError('unsupported arguments ' +
                             ', '.join(f'"{arg}"' for arg in unsupported_args))

        parameter_values = dict()
        for arg, param in self.parameters.items():
            try:
                parameter_values[arg] = param.validate(kwargs.get(arg, None))
            except ParameterError as error:
                raise ParameterError(f'invalid value for argument "{arg}": ' +
                                     error.args[0]) from None

        self.parameters = parameter_values

        self.setup()

    def __setattr__(self, name: str, value):
        if name in self.parameters:
            param = type(self).parameters[name]
            self.parameters[name] = param.validate(value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str):
        if name in self.parameters:
            return self.parameters[name]
        return super().__getattribute__(name)

    def setup(self) -> None:
        """Set up the benchmark before running."""
        pass

    @abc.abstractmethod
    def run(self) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Run the benchmark and return time."""
        pass

    def __call__(self) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        return self.run()

    def __repr__(self) -> str:
        return f'{type(self).__name__}(' + ', '.join(
            f'{param}={value!r}'
            for param, value in self.parameters.items()) + ')'
