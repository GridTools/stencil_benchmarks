import abc
import inspect

# pylint: disable=arguments-differ

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

    def __eq__(self, other):
        return (self.description == other.description
                and self.dtype == other.dtype and self.nargs == other.nargs)


class BenchmarkMeta(abc.ABCMeta):
    def __new__(cls, name, bases, namespace):
        if 'parameters' in namespace:
            raise AttributeError(
                'Benchmark classes must not define an attribute `parameters`')

        parameters = dict()
        transformed_attr_dict = dict()
        original_init = None
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, Parameter):
                parameters[attr_name] = attr_value
            elif attr_name == '__init__':
                original_init = attr_value
            else:
                transformed_attr_dict[attr_name] = attr_value

        if original_init is not None:
            if len(inspect.signature(original_init).parameters) != 1:
                raise ValueError(
                    '__init__ method must take only one argument (self)')

        def __init__(self, **kwargs):
            if parameters.keys() != kwargs.keys():
                required = set(parameters.keys())
                provided = set(kwargs.keys())

                if required - provided:
                    missing = ', '.join(f'“{arg}”'
                                        for arg in required - provided)
                    raise ValueError(
                        f'Missig values for required arguments {missing}')

                unexpected = ', '.join(f'“{arg}”'
                                       for arg in provided - required)
                raise ValueError(f'Unexpected arguments {unexpected}')
            for arg, value in kwargs.items():
                setattr(self, arg, parameters[arg].validate(value))
            if original_init is not None:
                original_init(self)

        transformed_attr_dict['__init__'] = __init__
        transformed_attr_dict['parameters'] = parameters
        benchmark = super().__new__(cls, name, bases, transformed_attr_dict)
        if not inspect.isabstract(benchmark):
            REGISTRY.add(benchmark)
        return benchmark


class Benchmark(metaclass=BenchmarkMeta):
    @abc.abstractmethod
    def run(self):
        """Run the benchmark and return time."""
