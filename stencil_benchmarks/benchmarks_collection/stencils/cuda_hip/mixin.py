import abc
import os
import warnings

import numpy as np

from .... import benchmark
from ....tools import array, cpphelpers, compilation, template
from . import api


class StencilMixin(benchmark.Benchmark):
    compiler = benchmark.Parameter('compiler path', 'nvcc')
    compiler_flags = benchmark.Parameter('compiler flags', '')
    backend = benchmark.Parameter('use NVIDIA CUDA or AMD HIP',
                                  'cuda',
                                  choices=['cuda', 'hip'])
    gpu_architecture = benchmark.Parameter('GPU architecture',
                                           dtype=str,
                                           nargs=1)
    print_code = benchmark.Parameter('print generated code', False)
    run_twice = benchmark.Parameter('run kernels twice and measure second run',
                                    False)
    gpu_timers = benchmark.Parameter(
        'use GPU timers instead of standard C++ timers', False)

    def setup(self):
        super().setup()

        template_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.template_file())
        code = template.render(template_file, **self.template_args())

        if self.print_code:
            print(cpphelpers.format_code(code))

        self.compiler_flags = (self.default_compiler_flags() + ' ' +
                               self.compiler_flags).strip()

        try:
            self.compiled = compilation.gnu_func(
                [self.compiler] + self.compiler_flags.split(), code, 'kernel',
                float)
        except compilation.CompilationError:
            raise benchmark.ParameterError('compilation failed')

        if self.verify and self.run_twice:
            warnings.warn(
                'enabling --run-twice and verification might lead to '
                'false negatives for stencils with read-write fields')

    def default_compiler_flags(self):
        flags = '-std=c++11 -DNDEBUG'
        if self.backend == 'cuda':
            if self.compiler.endswith('nvcc'):
                flags += ' -x cu -Xcompiler -Wall'
                if self.gpu_architecture:
                    flags += ' -arch ' + self.gpu_architecture
            elif self.compiler.endswith('clang++'):
                flags += ' -xcuda -Ofast -Wall -lcudart'
                if self.gpu_architecture:
                    flags += ' --cuda-gpu-arch=' + self.gpu_architecture
        elif self.backend == 'hip':
            flags += ' -xhip -Ofast -Wall'
            if self.gpu_architecture:
                flags += ' --amdgpu-target=' + self.gpu_architecture
        return flags

    @property
    def ctype_name(self):
        return compilation.dtype_cname(self.dtype)

    @property
    def sorted_domain(self):
        # pylint: disable=invalid-unary-operand-type
        indices = np.argsort(-np.array(self.strides))
        return tuple(np.array(self.domain)[indices])

    @property
    def sorted_strides(self):
        return tuple(sorted(self.strides, key=lambda x: -x))

    @property
    def sorted_block_size(self):
        # pylint: disable=invalid-unary-operand-type
        indices = np.argsort(-np.array(self.strides))
        return tuple(np.array(self.block_size)[indices])

    @abc.abstractmethod
    def template_file(self):
        pass

    @abc.abstractmethod
    def template_args(self):
        pass

    def run_stencil(self, data):
        offset = (self.halo, ) * 3
        device_data = [
            array.alloc_array(self.domain_with_halo,
                              self.dtype,
                              self.layout,
                              self.alignment,
                              index_to_align=offset,
                              alloc=self.runtime.malloc,
                              free=self.runtime.free,
                              apply_offset=self.offset_allocations)
            for _ in data
        ]
        for host_array, device_array in zip(data, device_data):
            self.runtime.memcpy(device_array.ctypes.data,
                                host_array.ctypes.data, host_array.nbytes,
                                'HostToDevice')
        self.runtime.device_synchronize()

        data_ptrs = [
            compilation.data_ptr(device_array, offset)
            for device_array in device_data
        ]

        time = self.compiled(*data_ptrs)
        if time == 0:
            raise benchmark.ExecutionError()

        if self.run_twice:
            time = self.compiled(*data_ptrs)

        for host_array, device_array in zip(data, device_data):
            self.runtime.memcpy(host_array.ctypes.data,
                                device_array.ctypes.data, host_array.nbytes,
                                'DeviceToHost')
        self.runtime.device_synchronize()
        return time

    @property
    def runtime(self):
        return api.runtime(self.backend)


class BasicStencilMixin(StencilMixin):
    loop = benchmark.Parameter('loop kind', '1D', choices=['1D', '3D'])
    block_size = benchmark.Parameter('block_size', (1024, 1, 1))

    def setup(self):
        super().setup()
        if self.loop == '1D' and sum(b != 1 for b in self.block_size) != 1:
            raise benchmark.ParameterError('block size must be 1 along '
                                           'all but one direction')

    @abc.abstractmethod
    def stencil_body(self):
        pass

    def template_file(self):
        return 'basic_' + self.loop.lower().replace("-", "_") + '.j2'

    def template_args(self):
        return dict(args=self.args,
                    ctype=self.ctype_name,
                    strides=self.strides,
                    sorted_strides=self.sorted_strides,
                    domain=self.domain,
                    sorted_domain=self.sorted_domain,
                    block_size=self.block_size,
                    sorted_block_size=self.sorted_block_size,
                    body=self.stencil_body(),
                    backend=self.backend,
                    gpu_timers=self.gpu_timers)


class VerticalAdvectionMixin(StencilMixin):
    block_size = benchmark.Parameter('block size', (32, 1))

    def template_file(self):
        return ('vertical_advection_' + type(self).__name__.lower() + '.j2')

    def template_args(self):
        return dict(args=self.args,
                    ctype=self.ctype_name,
                    strides=self.strides,
                    domain=self.domain,
                    block_size=self.block_size,
                    backend=self.backend,
                    gpu_timers=self.gpu_timers)


class HorizontalDiffusionMixin(StencilMixin):
    block_size = benchmark.Parameter('block size', (32, 8, 1))

    def template_file(self):
        return ('horizontal_diffusion_' + type(self).__name__.lower() + '.j2')

    def template_args(self):
        return dict(args=self.args,
                    ctype=self.ctype_name,
                    strides=self.strides,
                    domain=self.domain,
                    block_size=self.block_size,
                    backend=self.backend,
                    gpu_timers=self.gpu_timers)
