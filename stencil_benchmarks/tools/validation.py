import numpy as np


class ValidationError(RuntimeError):
    pass


def check_equality(result, expected):
    close = np.isclose(result, expected)
    if np.all(close):
        return
    failing_indices = np.nonzero(~close)
    n_report = 20
    n_failures = 0
    for n_failures, index in enumerate(zip(*failing_indices)):
        if n_failures < n_report:
            index_str = ', '.join(str(i) for i in index)
            print(f'failed at {index_str}: '
                  f'{result[index]:12.7f} != {expected[index]:12.7f}')
    if n_failures - n_report > 0:
        print(f'omitted further {n_failures - n_report} failures.')
    raise ValidationError()
