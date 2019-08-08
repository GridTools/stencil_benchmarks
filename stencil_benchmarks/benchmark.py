import abc
import inspect

# pylint: disable=arguments-differ,access-member-before-definition

REGISTRY = set()


class ValidationError(ValueError):
    pass


class Parameter:
    def __init__(self, description, default=None, dtype=None, nargs=None):
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

    def validate(self, value):
        if value is None:
            if self.default is None:
                raise ValidationError('value is required')
            return self.default

        if self.nargs == 1:
            if not isinstance(value, self.dtype):
                raise ValidationError('wrong type')
            return value

        if len(value) != self.nargs:
            raise ValidationError('wrong number of arguments')
        for val in value:
            if not isinstance(val, self.dtype):
                raise ValidationError('wrong type')
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
            except ValidationError as error:
                raise ValidationError(
                    f'validation of parameter "{arg}"" failed: ' +
                    error.args[0]) from None

        self.parameters = parameter_values

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
