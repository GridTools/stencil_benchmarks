import abc
import time

import numpy as np

from ... import benchmark

# pylint: disable=no-member


class Stencil(benchmark.Benchmark):
    domain = benchmark.Parameter('domain size', int, nargs=3)
    runs = benchmark.Parameter('number of runs', int)

    def setup(self):
        self.inputs = [
            np.random.uniform(size=self.domain) for _ in range(self.runs)
        ]
        self.outputs = [
            np.random.uniform(size=self.domain) for _ in range(self.runs)
        ]

    @abc.abstractmethod
    def apply(self, inp, out):
        pass

    def run(self):
        start = time.perf_counter()
        for run in range(self.runs):
            self.apply(self.inputs[run], self.outputs[run])
        stop = time.perf_counter()
        run_time = stop - start
        bandwidth = sum(
            data.nbytes
            for data in self.inputs + self.outputs) / run_time / 1e9
        return dict(time=run_time, bandwidth=bandwidth)


class Copy(Stencil):
    domain = benchmark.Parameter('domain size', int, nargs=3)
    runs = benchmark.Parameter('number of runs', int)

    def apply(self, inp, out):
        out[:] = inp[:]
