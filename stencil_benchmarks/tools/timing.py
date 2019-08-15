import functools
import time


def return_time(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        start = time.perf_counter()
        func(*args, **kwargs)
        stop = time.perf_counter()
        return stop - start

    return inner
