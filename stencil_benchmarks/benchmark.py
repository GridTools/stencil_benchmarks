import abc
import inspect

# pylint: disable=arguments-differ,access-member-before-definition

REGISTRY = set()


class Parameter:
    def __init__(self, description, dtype, nargs=1, default=None):
        self.description = description
        self.dtype = dtype
        self.nargs = nargs
        self.default = default

    def validate(self, value):
        if value is None:
            if self.default is None:
                raise ValueError('Value is required')
            return self.default

        if self.nargs == 1:
            if not isinstance(value, self.dtype):
                raise ValueError('Wrong type')
            return value

        if len(value) != self.nargs:
            raise ValueError('Wrong number of arguments')
        for val in value:
            if not isinstance(val, self.dtype):
                raise ValueError('Wrong type')
        return value

    def __repr__(self):
        return (f'{type(self).__name__}('
                f'description={self.description!r}, '
                f'dtype={self.dtype!r}, '
                f'nargs={self.nargs!r}, '
                f'default={self.default!r})')

    def __eq__(self, other):
        return (self.description == other.description
                and self.dtype == other.dtype and self.nargs == other.nargs)


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
        missing_args = set(self.parameters) - set(kwargs)
        if missing_args:
            raise ValueError('Missing values for arguments ' +
                             ', '.join(f'"{arg}"' for arg in missing_args))

        unsupported_args = set(kwargs) - set(self.parameters)
        if unsupported_args:
            raise ValueError('Unsupported arguments ' +
                             ', '.join(f'"{arg}"' for arg in unsupported_args))

        self.parameters = {
            arg: param.validate(kwargs[arg])
            for arg, param in self.parameters.items()
        }
        for arg, value in self.parameters.items():
            setattr(self, arg, value)
        self.setup()

    def setup(self):
        """Set up the benchmark before running."""

    @abc.abstractmethod
    def run(self):
        """Run the benchmark and return time."""

    def __repr__(self):
        return f'{type(self).__name__}(' + ', '.join(
            f'{param}={value!r}'
            for param, value in self.parameters.items()) + ')'
