import abc
import time

import numpy as np

from ...benchmark import Benchmark, Parameter
from ...tools import validation

# pylint: disable=abstract-method


class Stencil(Benchmark):
    domain = Parameter('domain size', int, nargs=3, default=(128, 128, 80))
    runs = Parameter('number of runs', int, default=1)
    data_sets = Parameter('number of data sets', int, default=1)
    halo = Parameter('halo size along horizontal dimensions', int, default=3)
    dtype = Parameter('data type', str, default='float64')
    verify = Parameter('enable verification', bool, default=True)

    def setup(self):
        super().setup()
        self.dtype = np.dtype(self.dtype)

    def _random_field(self):
        return np.random.uniform(-100, 100,
                                 size=self.domain_with_halo).astype(self.dtype)

    @property
    def domain_with_halo(self):
        return tuple(np.array(self.domain) + 2 * self.halo)

    def inner_slice(self, shift=None):
        if shift is None:
            shift = [0] * len(self.domain)
        return tuple(
            slice(self.halo + s, self.halo + d + s)
            for d, s in zip(self.domain, shift))

    @abc.abstractmethod
    def run_stencil(self, data_set):
        pass

    @abc.abstractmethod
    def verify_stencil(self, data_set):
        pass

    @abc.abstractproperty
    def data_size(self):
        pass

    def run(self):
        for run in range(self.data_sets):
            self.run_stencil(run)

        start = time.perf_counter()
        for run in range(self.runs):
            self.run_stencil(run % self.data_sets)
        stop = time.perf_counter()

        if self.verify:
            for data_set in range(self.data_sets):
                self.verify_stencil(data_set)

        run_time = stop - start
        bandwidth = self.data_size * self.runs / run_time / 1e9
        return {'time': run_time, 'bandwidth': bandwidth}


class BasicStencil(Stencil):
    def setup(self):
        super().setup()

        self.inouts = [(self._random_field(), self._random_field())
                       for _ in range(self.data_sets)]

    @property
    def data_size(self):
        return 2 * np.product(self.domain) * self.dtype.itemsize


class CopyStencil(BasicStencil):
    def verify_stencil(self, data_set):
        inp, out = self.inouts[data_set]
        validation.check_equality(out[self.inner_slice()],
                                  inp[self.inner_slice()])


class OnesidedAverageStencil(BasicStencil):
    axis = Parameter('axis along which to average', int, default=0)

    def setup(self):
        super().setup()
        if not 0 <= self.axis <= 2:
            raise ValueError('invalid axis')

    def verify_stencil(self, data_set):
        inp, out = self.inouts[data_set]
        shift = np.zeros(3, dtype=int)
        shift[self.axis] = 1
        validation.check_equality(
            out[self.inner_slice()],
            (inp[self.inner_slice(shift)] + inp[self.inner_slice()]) / 2)


class SymmetricAverageStencil(BasicStencil):
    axis = Parameter('axis along which to average', int, default=0)

    def setup(self):
        super().setup()
        if not 0 <= self.axis <= 2:
            raise ValueError('invalid axis')

    def verify_stencil(self, data_set):
        inp, out = self.inouts[data_set]
        shift = np.zeros(3, dtype=int)
        shift[self.axis] = 1
        validation.check_equality(
            out[self.inner_slice()],
            (inp[self.inner_slice(shift)] + inp[self.inner_slice(-shift)]) / 2)


class LaplacianStencil(BasicStencil):
    along_x = Parameter('include x-axis in Laplacian', bool, default=True)
    along_y = Parameter('include y-axis in Laplacian', bool, default=True)
    along_z = Parameter('include z-axis in Laplacian', bool, default=False)

    def setup(self):
        super().setup()
        if self.halo < 1:
            raise ValueError(f'positive halo size required')

    def verify_stencil(self, data_set):
        inp, out = self.inouts[data_set]
        result = np.zeros(self.domain)
        along_axes = (self.along_x, self.along_y, self.along_z)
        for axis, apply_along_axis in enumerate(along_axes):
            if apply_along_axis:
                shift = np.zeros(3, dtype=int)
                shift[axis] = 1
                result += (2 * inp[self.inner_slice()] -
                           inp[self.inner_slice(shift)] -
                           inp[self.inner_slice(-shift)])
        validation.check_equality(out[self.inner_slice()], result)
