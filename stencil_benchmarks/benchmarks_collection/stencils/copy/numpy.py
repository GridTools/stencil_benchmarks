import time

import numpy as np

from .... import benchmark


class Numpy(benchmark.Benchmark):
    domain = benchmark.Parameter('domain size', int, nargs=3)
    runs = benchmark.Parameter('number of runs', int)

    def __init__(self):
        self.inputs = [
            np.random.uniform(size=self.domain) for _ in range(self.runs)
        ]
        self.outputs = [
            np.random.uniform(size=self.domain) for _ in range(self.runs)
        ]

    def run(self):
        start = time.perf_counter()
        for run in range(self.runs):
            self.outputs[run][:] = self.inputs[run][:]
        stop = time.perf_counter()

        bandwidth = sum(
            data.nbytes
            for data in self.inputs + self.outputs) / (stop - start) / 1e9
        return dict(time=stop - start, bandwidth=bandwidth)
