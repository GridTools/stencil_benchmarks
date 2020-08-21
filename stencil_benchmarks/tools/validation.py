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
import click
import numpy as np


class ValidationError(RuntimeError):
    pass


def _report_failures_large(result, expected):
    failing_indices = np.nonzero(~np.isclose(result, expected))
    n_report = 20
    n_failures = 0
    for n_failures, index in enumerate(zip(*failing_indices)):
        if n_failures < n_report:
            index_str = ', '.join(str(i) for i in index)
            print(f'failed at {index_str}: '
                  f'{result[index]:12.7f} != {expected[index]:12.7f}')
    if n_failures - n_report > 0:
        print(f'omitted further {n_failures - n_report} failures.')


def _report_failures_small(result, expected):
    assert result.ndim == expected.ndim == 3

    def print_slice(data, correct):
        fmt = '{:12.6g}'
        for j in reversed(range(data.shape[1])):
            for i in range(data.shape[0]):
                click.echo(click.style(fmt.format(data[i, j]),
                                       fg='green' if correct[i, j] else 'red'),
                           nl=False)
            click.echo()

    for k in range(expected.shape[2]):
        correct = np.isclose(result[:, :, k], expected[:, :, k])
        print(f'result[:, :, {k}]:')
        print_slice(result[:, :, k], correct)
        print(f'expected[:, :, {k}]:')
        print_slice(expected[:, :, k], correct)


def check_equality(result, expected):
    close = np.isclose(result, expected)
    if np.all(close):
        return
    if result.ndim != 3 or np.product(result.shape) > 1000:
        _report_failures_large(result, expected)
    else:
        _report_failures_small(result, expected)
    raise ValidationError()
