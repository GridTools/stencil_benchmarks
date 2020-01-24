import abc
import collections
import copy

import numpy as np

from ...benchmark import Benchmark, Parameter, ParameterError
from ...tools import alloc, array, validation

# pylint: disable=abstract-method


class Stencil(Benchmark):
    domain = Parameter('domain size', (128, 128, 80))
    data_sets = Parameter(
        'number of data sets, if bigger than one, data sets are cycled before '
        'each execution to start with cold cache', 1)
    halo = Parameter('halo size', 3)
    dtype = Parameter('data type in NumPy format, e.g. float32 or float64',
                      'float64')
    layout = Parameter('data layout, 2 means innermost dimension, 0 outermost',
                       (2, 1, 0))
    alignment = Parameter('data alignment in bytes', 0)
    huge_pages = Parameter('use huge pages', False)
    offset_allocations = Parameter(
        'offset allocated data by some bytes to minimize cache conflicts',
        False)
    verify = Parameter('enable verification', True)

    def setup(self):
        super().setup()
        if self.halo < 0:
            raise ParameterError(f'negative halo size given ({self.halo}')
        if tuple(sorted(self.layout)) != (0, 1, 2):
            raise ParameterError(f'invalid layout specification {self.layout}')
        if self.alignment < 0:
            raise ParameterError(
                f'negative alignment given ({self.alignment} bytes)')
        if self.alignment % self.dtype_size != 0:
            raise ParameterError(
                f'alignment ({self.alignment} bytes) not divisible '
                f'by dtype size ({self.dtype_size} bytes)')

        stencil_data = collections.namedtuple('StencilData', self.args)
        self._data = [
            stencil_data._make(self.random_field()
                               for _ in range(len(self.args)))
            for _ in range(self.data_sets)
        ]
        self._run = 0

    def empty_field(self):
        def alloc_data(nbytes):
            return alloc.mmap(nbytes, self.huge_pages)

        return array.alloc_array(self.domain_with_halo,
                                 self.dtype,
                                 self.layout,
                                 self.alignment,
                                 index_to_align=(self.halo, ) * 3,
                                 alloc=alloc_data,
                                 apply_offset=self.offset_allocations)

    def random_field(self):
        data = self.empty_field()
        data[:, :, :] = np.random.uniform(size=self.domain_with_halo).astype(
            self.dtype)
        return data

    @property
    def dtype_size(self):
        return np.dtype(self.dtype).itemsize

    @property
    def domain_with_halo(self):
        return tuple(np.array(self.domain) + 2 * self.halo)

    @property
    def strides(self):
        return tuple(np.array(self._data[0][0].strides) // self.dtype_size)

    @property
    def data_size(self):
        return len(self.args) * np.product(self.domain) * self.dtype_size

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
        data_index = self._run % self.data_sets

        if self.verify:
            data_before = copy.deepcopy(self._data[data_index])

        run_time = self.run_stencil(self._data[data_index])

        if self.verify:
            self.verify_stencil(data_before, self._data[data_index])

        self._run += 1
        assert run_time > 0
        bandwidth = self.data_size / run_time / 1e9
        return {'time': run_time, 'bandwidth': bandwidth}


class BasicStencil(Stencil):
    @property
    def args(self):
        return 'inp', 'out'


class EmptyStencil(BasicStencil):
    def verify_stencil(self, data_before, data_after):
        pass


class CopyStencil(BasicStencil):
    def verify_stencil(self, data_before, data_after):
        validation.check_equality(data_before.inp, data_after.inp)

        validation.check_equality(data_after.out[self.inner_slice()],
                                  data_before.inp[self.inner_slice()])


class OnesidedAverageStencil(BasicStencil):
    axis = Parameter('axis along which to average', 0, choices=[0, 1, 2])

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
    axis = Parameter('axis along which to average', 0, choices=[0, 1, 2])

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
            raise ParameterError(
                f'positive halo size required (given halo: {self.halo})')

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
            raise ParameterError(
                f'halo size must be at least 2 (given halo: {self.halo})')

    @property
    def args(self):
        return 'inp', 'coeff', 'out'

    @property
    def data_size(self):
        return (2 * np.product(self.domain) +
                np.product(np.array(self.domain) + 4)) * self.dtype_size

    def verify_stencil(self, data_before, data_after):
        validation.check_equality(data_before.inp, data_after.inp)
        validation.check_equality(data_before.coeff, data_after.coeff)

        inp = data_before.inp
        coeff = data_before.coeff
        out = data_after.out

        lap = np.zeros_like(inp)
        lap[1:-1, 1:-1, :] = 4 * inp[1:-1, 1:-1, :] - (
            inp[2:, 1:-1, :] + inp[:-2, 1:-1, :] + inp[1:-1, 2:, :] +
            inp[1:-1, :-2, :])

        flx = np.zeros_like(inp)
        flx[:-1, :, :] = lap[1:, :, :] - lap[:-1, :, :]
        flx[:-1, :, :] = np.where(
            flx[:-1, :, :] * (inp[1:, :, :] - inp[:-1, :, :]) > 0, 0,
            flx[:-1, :, :])

        fly = np.zeros_like(inp)
        fly[:, :-1, :] = lap[:, 1:, :] - lap[:, :-1, :]
        fly[:, :-1, :] = np.where(
            fly[:, :-1, :] * (inp[:, 1:, :] - inp[:, :-1, :]) > 0, 0,
            fly[:, :-1, :])

        result = np.zeros_like(inp)
        result[1:-1, 1:-1, :] = inp[1:-1, 1:-1, :] - coeff[1:-1, 1:-1, :] * (
            flx[1:-1, 1:-1, :] - flx[:-2, 1:-1, :] + fly[1:-1, 1:-1, :] -
            fly[1:-1, :-2, :])

        validation.check_equality(out[self.inner_slice()],
                                  result[self.inner_slice()])


class VerticalAdvectionStencil(Stencil):
    def setup(self):
        super().setup()

        if self.halo < 1:
            raise ParameterError(
                f'positive halo size required (given halo: {self.halo})')

    @property
    def args(self):
        return ('ustage', 'upos', 'utens', 'utensstage', 'vstage', 'vpos',
                'vtens', 'vtensstage', 'wstage', 'wpos', 'wtens', 'wtensstage',
                'wcon', 'ccol', 'dcol', 'datacol')

    @property
    def data_size(self):
        reads = 16  # including ccol + dcol, but not datacol
        writes = 5  # including ccol + dcol, but not datacol
        return (reads + writes) * np.product(self.domain) * self.dtype_size

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
