import contextlib

import click


@contextlib.contextmanager
def report_progress(total):
    current = 0

    def report():
        nonlocal current
        percent = min(100 * current // total, 100)
        click.echo('#' * percent + '-' * (100 - percent) + f' {percent:3}%\r',
                   nl=False)
        current += 1

    report()
    yield report
    click.echo(' ' * 110 + '\r', nl=False)
