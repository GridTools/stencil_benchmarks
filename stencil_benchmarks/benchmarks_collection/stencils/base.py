import abc
import collections
import copy
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

        stencil_data = collections.namedtuple('StencilData', self.args)
        self._data = [
            stencil_data._make(self.random_field()
                               for _ in range(len(self.args)))
            for _ in range(self.data_sets)
        ]

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

    @property
    def strides(self):
        return tuple(np.array(self._data[0][0].strides) // self.dtype.itemsize)

    @property
    def data_size(self):
        return len(self.args) * np.product(self.domain) * self.dtype.itemsize

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
    def run_stencil(self, data):
        pass

    @abc.abstractmethod
    def verify_stencil(self, data_before, data_after):
        pass

    @abc.abstractproperty
    def args(self):
        pass

    def run(self):
        if self.verify:
            for data_set in range(self.data_sets):
                data_before = copy.deepcopy(self._data[data_set])
                self.run_stencil(self._data[data_set])
                data_after = self._data[data_set]
                self.verify_stencil(data_before, data_after)

        start = time.perf_counter()
        for run in range(self.runs):
            self.run_stencil(self._data[run % self.data_sets])
        stop = time.perf_counter()

        run_time = stop - start
        bandwidth = self.data_size * self.runs / run_time / 1e9
        return {'time': run_time, 'bandwidth': bandwidth}


class BasicStencil(Stencil):
    @property
    def args(self):
        return 'inp', 'out'


class CopyStencil(BasicStencil):
    def verify_stencil(self, data_before, data_after):
        validation.check_equality(data_before.inp, data_after.inp)

        validation.check_equality(data_after.out[self.inner_slice()],
                                  data_before.inp[self.inner_slice()])


class OnesidedAverageStencil(BasicStencil):
    axis = Parameter('axis along which to average', 0)

    def setup(self):
        super().setup()
        if not 0 <= self.axis <= 2:
            raise ValueError('invalid axis')

    def verify_stencil(self, data_before, data_after):
        validation.check_equality(data_before.inp, data_after.inp)

        inp = data_before.inp
        out = data_after.out
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

    def verify_stencil(self, data_before, data_after):
        validation.check_equality(data_before.inp, data_after.inp)

        inp = data_before.inp
        out = data_after.out
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

    def verify_stencil(self, data_before, data_after):
        validation.check_equality(data_before.inp, data_after.inp)

        inp = data_before.inp
        out = data_after.out
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

    @property
    def args(self):
        return 'inp', 'coeff', 'out'

    def verify_stencil(self, data_before, data_after):
        validation.check_equality(data_before.inp, data_after.inp)
        validation.check_equality(data_before.coeff, data_after.coeff)

        inp = data_before.inp
        coeff = data_before.coeff
        out = data_after.out

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


class VerticalAdvectionStencil(Stencil):
    def setup(self):
        super().setup()

        if self.halo < 1:
            raise ValueError('halo size must be at least 1')

    @property
    def args(self):
        return ('ustage', 'upos', 'utens', 'utensstage', 'vstage', 'vpos',
                'vtens', 'vtensstage', 'wstage', 'wpos', 'wtens', 'wtensstage',
                'wcon', 'ccol', 'dcol', 'datacol')

    @property
    def data_size(self):
        return 16 * np.product(self.domain) * self.dtype.itemsize

    def verify_stencil(self, data_before, data_after):
        # pylint: disable=unsubscriptable-object

        validation.check_equality(data_before.ustage, data_after.ustage)
        validation.check_equality(data_before.upos, data_after.upos)
        validation.check_equality(data_before.utens, data_after.utens)
        validation.check_equality(data_before.vstage, data_after.vstage)
        validation.check_equality(data_before.vpos, data_after.vpos)
        validation.check_equality(data_before.vtens, data_after.vtens)
        validation.check_equality(data_before.wstage, data_after.wstage)
        validation.check_equality(data_before.wpos, data_after.wpos)
        validation.check_equality(data_before.wtens, data_after.wtens)
        validation.check_equality(data_before.wcon, data_after.wcon)

        halo = self.halo
        domain = self.domain

        class Wrapper:
            def __init__(self, data):
                self.data = data

            def __getitem__(self, index):
                if isinstance(index, tuple):
                    i, j, k = index
                else:
                    i, j, k = 0, 0, index
                data_slice = (slice(halo + i, halo + domain[0] + i),
                              slice(halo + j, halo + domain[1] + j), halo + k)
                return self.data[data_slice]

            def __setitem__(self, index, value):
                self.__getitem__(index)[:] = value

        (ustage, upos, utens, utensstage, vstage, vpos, vtens, vtensstage,
         wstage, wpos, wtens, wtensstage, wcon, ccol, dcol,
         datacol) = [Wrapper(data) for data in data_before]

        dtr_stage = 3 / 20
        beta_v = 0
        bet_m = 0.5 * (1 - beta_v)
        bet_p = 0.5 * (1 + beta_v)

        def forward_sweep(ishift, jshift, stage, pos, tens, tensstage):
            # pylint: disable=invalid-name
            k = 0
            gcv = 0.25 * (wcon[ishift, jshift, k + 1] + wcon[k + 1])

            cs = gcv * bet_m

            ccol[k] = gcv * bet_p
            bcol = dtr_stage - ccol[k]

            correction_term = -cs * (stage[k + 1] - stage[k])
            dcol[k] = dtr_stage * pos[k] + tens[k] + tensstage[
                k] + correction_term

            ccol[k] /= bcol
            dcol[k] /= bcol

            for k in range(1, domain[2] - 1):
                gav = -0.25 * (wcon[ishift, jshift, k] + wcon[k])
                gcv = 0.25 * (wcon[ishift, jshift, k + 1] + wcon[k + 1])

                as_ = gav * bet_m
                cs = gcv * bet_m

                acol = gav * bet_p
                ccol[k] = gcv * bet_p
                bcol = dtr_stage - acol - ccol[k]

                correction_term = (-as_ * (stage[k - 1] - stage[k]) - cs *
                                   (stage[k + 1] - stage[k]))
                dcol[k] = (dtr_stage * pos[k] + tens[k] + tensstage[k] +
                           correction_term)

                divided = 1.0 / (bcol - ccol[k - 1] * acol)
                ccol[k] *= divided
                dcol[k] = (dcol[k] - dcol[k - 1] * acol) * divided

            k = domain[2] - 1
            gav = -0.25 * (wcon[ishift, jshift, k] + wcon[k])

            as_ = gav * bet_m

            acol = gav * bet_p
            bcol = dtr_stage - acol

            correction_term = -as_ * (stage[k - 1] - stage[k])
            dcol[k] = (dtr_stage * pos[k] + tens[k] + tensstage[k] +
                       correction_term)

            dcol[k] = ((dcol[k] - dcol[k - 1] * acol) /
                       (bcol - ccol[k - 1] * acol))

        def backward_sweep(pos, tensstage):
            k = domain[2] - 1
            datacol[k] = dcol[k]
            ccol[k] = datacol[k]
            tensstage[k] = dtr_stage * (datacol[k] - pos[k])

            for k in range(domain[2] - 2, -1, -1):
                datacol[k] = dcol[k] - ccol[k] * datacol[k + 1]
                ccol[k] = datacol[k]
                tensstage[k] = dtr_stage * (datacol[k] - pos[k])

        forward_sweep(1, 0, ustage, upos, utens, utensstage)
        backward_sweep(upos, utensstage)

        forward_sweep(0, 1, vstage, vpos, vtens, vtensstage)
        backward_sweep(vpos, vtensstage)

        forward_sweep(0, 0, wstage, wpos, wtens, wtensstage)
        backward_sweep(wpos, wtensstage)

        validation.check_equality(data_before.utensstage,
                                  data_after.utensstage)
        validation.check_equality(data_before.vtensstage,
                                  data_after.vtensstage)
        validation.check_equality(data_before.wtensstage,
                                  data_after.wtensstage)
