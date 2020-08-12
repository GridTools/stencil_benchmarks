import abc
import contextlib
import ctypes
import os
import warnings

from stencil_benchmarks.benchmark import (Benchmark, ExecutionError, Parameter,
                                          ParameterError)
from stencil_benchmarks.tools import array, cpphelpers, compilation, template
from . import api


class StencilMixin(Benchmark):
    compiler = Parameter('compiler path', 'nvcc')
    compiler_flags = Parameter('compiler flags', '')
    backend = Parameter('use NVIDIA CUDA or AMD HIP',
                        'cuda',
                        choices=['cuda', 'hip'])
    gpu_architecture = Parameter('GPU architecture', dtype=str, nargs=1)
    print_code = Parameter('print generated code', False)
    dry_runs = Parameter('kernel dry-runs before the measurement', 0)
    gpu_timers = Parameter('use GPU timers instead of standard C++ timers',
                           False)

    def setup(self):
        super().setup()

        template_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'templates',
            self.template_file())
        code = template.render(template_file, **self.template_args())

        if self.print_code:
            print(cpphelpers.format_code(code))

        self.compiler_flags = (self.default_compiler_flags() + ' ' +
                               self.compiler_flags).strip()

        try:
            self.compiled = compilation.GnuLibrary(code, [self.compiler] +
                                                   self.compiler_flags.split())
        except compilation.CompilationError as error:
            raise ParameterError(*error.args) from error

        if self.verify and self.dry_runs:
            warnings.warn(
                'using --dry-runs together with verification might lead to '
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

    @abc.abstractmethod
    def template_file(self):
        pass

    def template_args(self):
        return dict(args=self.args,
                    backend=self.backend,
                    ctype=compilation.dtype_cname(self.dtype),
                    domain=self.domain,
                    dry_runs=self.dry_runs,
                    gpu_timers=self.gpu_timers,
                    strides=self.strides)

    @contextlib.contextmanager
    def on_device(self, data):
        runtime = api.runtime(self.backend)

        device_data = [
            array.alloc_array(self.domain_with_halo,
                              self.dtype,
                              self.layout,
                              self.alignment,
                              index_to_align=(self.halo, ) * 3,
                              alloc=runtime.malloc,
                              apply_offset=self.offset_allocations)
            for _ in data
        ]

        for host_array, device_array in zip(data, device_data):
            runtime.memcpy(device_array.ctypes.data, host_array.ctypes.data,
                           array.nbytes(host_array), 'HostToDevice')
        runtime.device_synchronize()

        yield device_data

        for host_array, device_array in zip(data, device_data):
            runtime.memcpy(host_array.ctypes.data, device_array.ctypes.data,
                           array.nbytes(host_array), 'DeviceToHost')
        runtime.device_synchronize()

    def run_stencil(self, data):
        with self.on_device(data) as device_data:
            data_ptrs = [
                compilation.data_ptr(device_array, (self.halo, ) * 3)
                for device_array in device_data
            ]

            time = ctypes.c_double()
            try:
                self.compiled.kernel(ctypes.byref(time), *data_ptrs)
            except compilation.ExecutionError as error:
                raise ExecutionError() from error

        return time.value
