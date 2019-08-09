import abc
import time

import numpy as np

from ...benchmark import Benchmark, Parameter
from ...tools import array, validation

# pylint: disable=abstract-method


class Stencil(Benchmark):
    domain = Parameter('domain size', (128, 128, 80))
    runs = Parameter('number of runs', 1)
    data_sets = Parameter('number of data sets', 1)
    halo = Parameter('halo size along horizontal dimensions', 3)
    dtype = Parameter('data type', 'float64')
    layout = Parameter('data layout', (2, 1, 0))
    alignment = Parameter('data alignment (in bytes)', 0)
    verify = Parameter('enable verification', True)

    def setup(self):
        super().setup()
        self.dtype = np.dtype(self.dtype)
        if tuple(sorted(self.layout)) != (0, 1, 2):
            raise ValueError('invalid layout specification')
        if self.alignment % self.dtype.itemsize != 0:
            raise ValueError('alignment not divisible by dtype itemsize')

    def empty_field(self):
        return array.alloc_array(self.domain_with_halo,
                                 self.dtype,
                                 self.layout,
                                 self.alignment,
                                 index_to_align=(self.halo, self.halo,
                                                 self.halo))

    def random_field(self):
        data = self.empty_field()
        data[:, :, :] = np.random.uniform(size=self.domain_with_halo).astype(
            self.dtype)
        return data

    @property
    def domain_with_halo(self):
        return tuple(np.array(self.domain) + 2 * self.halo)

    def inner_slice(self, shift=None, expand=None):
        if shift is None:
            shift = [0] * len(self.domain)
        if expand is None:
            expand = [0] * len(self.domain)
        elif isinstance(expand, int):
            expand = [expand] * len(self.domain)
        return tuple(
            slice(self.halo + s - e, self.halo + d + s + e)
            for d, s, e in zip(self.domain, shift, expand))

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

        self.inouts = [(self.random_field(), self.random_field())
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
    axis = Parameter('axis along which to average', 0)

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
    axis = Parameter('axis along which to average', 0)

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
    along_x = Parameter('include x-axis in Laplacian', True)
    along_y = Parameter('include y-axis in Laplacian', True)
    along_z = Parameter('include z-axis in Laplacian', False)

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


class HorizontalDiffusionStencil(Stencil):
    def setup(self):
        super().setup()

        if self.halo < 2:
            raise ValueError('halo size must be at least 2')

        self.inouts = [(self.random_field(), self.random_field(),
                        self.random_field()) for _ in range(self.data_sets)]

    @property
    def data_size(self):
        return 3 * np.product(self.domain) * self.dtype.itemsize

    def verify_stencil(self, data_set):
        inp, coeff, out = self.inouts[data_set]

        lap = 4 * inp[1:-1, 1:-1, :] - (inp[2:, 1:-1, :] + inp[:-2, 1:-1, :] +
                                        inp[1:-1, 2:, :] + inp[1:-1, :-2, :])

        flx = lap[1:, :-1, :] - lap[:-1, :-1, :]
        flx = np.where(flx * (inp[2:-1, 1:-2, :] - inp[1:-2, 1:-2, :]) > 0, 0,
                       flx)

        fly = lap[:-1, 1:, :] - lap[:-1, :-1, :]
        fly = np.where(fly * (inp[1:-2, 2:-1, :] - inp[1:-2, 1:-2, :]) > 0, 0,
                       fly)

        result = np.empty_like(out)
        result[2:-2, 2:-2, :] = inp[2:-2, 2:-2, :] - coeff[2:-2, 2:-2, :] * (
            flx[1:, 1:, :] - flx[:-1, 1:, :] + fly[1:, 1:, :] -
            fly[1:, :-1, :])

        validation.check_equality(out[self.inner_slice()],
                                  result[self.inner_slice()])
