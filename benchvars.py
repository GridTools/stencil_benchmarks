import enum
import functools
import itertools

class VarType(enum.Enum):
    compiletime = 1
    runtime = 2
    environment = 3

class Var():
    def __init__(self, var_options, value):
        self.var_options = var_options
        self.value = value

    def __str__(self):
        return self.var_options.fmt(self.value)

    def __repr__(self):
        return str(self)

    def short_str(self):
        return self.var_options.sfmt(self.value)

    @property
    def name(self):
        return self.var_options.name


class VarOptions():
    def __init__(self, type, name, values, fmt=None, sfmt=None):
        self.type = type
        self.name = name
        self.values = values
        if fmt is None:
            if type is VarType.runtime:
                self.fmt = lambda v: '--{} {}'.format(self.name, v)
            else:
                self.fmt = lambda v: '{}={}'.format(self.name, v)
        else:
            self.fmt = fmt
        if sfmt is None:
            self.sfmt = lambda v: '{}{}'.format(self.name[0].lower(), v)
        else:
            self.sfmt = sfmt

    def generate(self):
        for value in self.values:
            yield Var(self, value)

    def __str__(self):
        return 'Options for {}: '.format(self.name) + ', '.join(str(v) for v in self.values)

    def __repr__(self):
        return str(self)


class VarOptionsList():
    def __init__(self):
        self.var_options = []

    def add(self, *args, **kwargs):
        self.var_options.append(VarOptions(*args, **kwargs))
        return self

    def add_rt(self, *args, **kwargs):
        return self.add(VarType.runtime, *args, **kwargs)

    def add_ct(self, *args, **kwargs):
        return self.add(VarType.compiletime, *args, **kwargs)

    def add_env(self, *args, **kwargs):
        return self.add(VarType.environment, *args, **kwargs)

    def by_type(self, type):
        if type is None:
            return self.var_options
        else:
            return [vo for vo in self.var_options if vo.type is type]

    def __str__(self):
        return ('\n'.join(str(vo) for vo in self.var_options)
                + '\nTotal options: {} '.format(self.total_options())
                + '(CT: {}, RT: {}, Env: {})'.format(self.total_options(VarType.compiletime),
                                                     self.total_options(VarType.runtime),
                                                     self.total_options(VarType.environment)))

    def generate(self, type=None):
        return itertools.product(*(vo.generate() for vo in self.by_type(type)))

    def total_options(self, type=None):
        return functools.reduce(lambda a, b: a * b, (len(vo.values) for vo in self.by_type(type)))

