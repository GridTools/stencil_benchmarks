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
