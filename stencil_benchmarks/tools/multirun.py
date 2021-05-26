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
from ast import literal_eval
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


def default_kwargs(**kwargs):
    def override(options=None, **overrides):
        kws = kwargs.copy()
        kws.update(overrides)
        for o in options:
            name, value = o.split('=', 1)
            name = name.replace('-', '_')
            value = literal_eval(value)
            kws[name] = value
        return kws

    return override
