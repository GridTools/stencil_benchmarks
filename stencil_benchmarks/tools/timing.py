import functools
import time
from typing import Callable


def return_time(func: Callable[..., None]) -> Callable[..., float]:
    @functools.wraps(func)
    def inner(*args, **kwargs):
        start = time.perf_counter()
        func(*args, **kwargs)
        stop = time.perf_counter()
        return stop - start

    return inner
