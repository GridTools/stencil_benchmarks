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


def _report_failures_large(result, expected, failure):
    failing_indices = np.argwhere(failure)
    n_report = 20
    n_failures = len(failing_indices)
    report = ''
    for index in failing_indices[:n_report]:
        index_str = ', '.join(f'{i:3}' for i in index) + ':'
        report += (f'failed at {index_str} {result[tuple(index)]:12.7f}    != '
                   f'{expected[tuple(index)]:12.7f}\n')
    if n_failures - n_report > 0:
        report += f'omitted further {n_failures - n_report} failures.'
    return report


def _report_failures_small(result, expected, failure):
    assert result.ndim == expected.ndim == 3

    report = ''

    def print_slice(data, failure):
        nonlocal report
        fmt = '{:12.7f}'
        for j in reversed(range(data.shape[1])):
            for i in range(data.shape[0]):
                report += click.style(fmt.format(data[i, j]),
                                      fg='red' if failure[i, j] else 'green')
            report += '\n'
        return report

    matching = None

    def print_matching():
        nonlocal matching, report
        if matching:
            match_str = f'{matching[0]}'
            if matching[1] - matching[0] > 1:
                match_str = f'{match_str}:{matching[1]}'
            report += f'no failures in result[:, :, {match_str}]\n'
        matching = None

    for k in range(expected.shape[2]):
        if np.any(failure[:, :, k]):
            print_matching()
            report += f'result[:, :, {k}]:\n'
            print_slice(result[:, :, k], failure[:, :, k])
            report += f'expected[:, :, {k}]:\n'
            print_slice(expected[:, :, k], failure[:, :, k])
        else:
            matching = (matching[0], k + 1) if matching else (k, k + 1)
    print_matching()

    return report


def _tolerances(dtype):
    if dtype == np.float32:
        return dict(rtol=1e-4, atol=1e-5)
    return dict()


def check_equality(name, result, expected):
    assert result.dtype == expected.dtype
    failure = ~np.isclose(result, expected, **_tolerances(result.dtype))
    if not np.any(failure):
        return
    if result.ndim != 3 or np.product(result.shape) > 1000:
        report = _report_failures_large(result, expected, failure)
    else:
        report = _report_failures_small(result, expected, failure)
    raise ValidationError(f'validation of field {name} failed at '
                          f'{np.count_nonzero(failure)} points:\n' + report)
