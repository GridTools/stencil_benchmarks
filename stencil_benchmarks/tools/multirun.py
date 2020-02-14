import gc
import re

import pandas as pd

from . import cli


class Configuration:
    def __init__(self, ctor, name=None, **kwargs):
        self.ctor = ctor
        if name is None:
            self.name = re.sub('(?!^)([A-Z1-9]+)', r'-\1',
                               ctor.__name__).lower()
        else:
            self.name = name
        self.kwargs = kwargs

    def __call__(self, preprocess_args=None, **kwargs):
        if preprocess_args is None:

            def preprocess_args(**kwargs):
                return kwargs

        run = self.ctor(**preprocess_args(**self.kwargs, **kwargs))

        def modified_run():
            result = run()
            result.update(name=self.name)
            result.update(cli.pretty_parameters(run))
            return result

        return modified_run


def domains(min_exp=5, max_exp=11, k=80):
    for exponent in range(min_exp, max_exp):
        d = 2**exponent
        yield d, d, k


def run_scaling_benchmark(configurations,
                          executions,
                          preprocess_args=None,
                          domain_range=None):
    if domain_range is None:
        domain_range = domains()

    results = []
    with cli.ProgressBar() as progress:
        for domain in progress.report(domain_range):
            for config in progress.report(configurations):
                run = config(preprocess_args=preprocess_args, domain=domain)
                results += [run() for _ in progress.report(range(executions))]
                del run
                gc.collect()
    return pd.DataFrame(results)


def truncate_block_size_to_domain(**kwargs):
    if 'block_size' in kwargs and 'domain' in kwargs:
        kwargs['block_size'] = tuple(
            min(b, d) for b, d in zip(kwargs['block_size'], kwargs['domain']))
    return kwargs
