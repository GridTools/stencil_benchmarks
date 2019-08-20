import abc
import ctypes
import os

import numpy as np

from .... import benchmark
from ....tools import array, cpphelpers, compilation, template


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

    def setup(self):
        super().setup()

        template_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.template_file())
        code = template.render(template_file, **self.template_args())

        if self.print_code:
            print(cpphelpers.format_code(code))

        self.compiled = compilation.gnu_func(self.compile_command(), code,
                                             'kernel', float)

    def compile_command(self):
        command = [self.compiler]
        command += ['-std=c++11', '-DNDEBUG']
        if self.backend == 'cuda':
            if self.compiler.endswith('nvcc'):
                command += ['-x', 'cu', '-Xcompiler', '-Wall']
                if self.gpu_architecture:
                    command += ['-arch', self.gpu_architecture]
            elif self.compiler.endswith('clang++'):
                command += ['-xcuda', '-Ofast', '-Wall', '-lcudart']
                if self.gpu_architecture:
                    command += ['--cuda-gpu-arch=' + self.gpu_architecture]
        elif self.backend == 'hip':
            command += ['-xhip', '-Ofast', '-Wall']
            if self.gpu_architecture:
                command += ['--amdgpu-target=' + self.gpu_architecture]
        if self.compiler_flags:
            command += self.compiler_flags.split()
        return command

    @property
    def ctype_name(self):
        return compilation.ctype_cname(compilation.dtype_as_ctype(self.dtype))

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
                              alloc=self.alloc,
                              free=self.free,
                              apply_offset=self.offset_allocations)
            for _ in data
        ]
        for host_array, device_array in zip(data, device_data):
            self.memcpy(device_array.ctypes.data, host_array.ctypes.data,
                        host_array.nbytes, 'HostToDevice')

        time = self.compiled(*(compilation.data_ptr(device_array, offset)
                               for device_array in device_data))

        for host_array, device_array in zip(data, device_data):
            self.memcpy(host_array.ctypes.data, device_array.ctypes.data,
                        host_array.nbytes, 'DeviceToHost')
        return time

    @property
    def runtime(self):
        if not hasattr(self, '_runtime'):
            if self.backend == 'cuda':
                lib = 'libcudart.so'
            elif self.backend == 'hip':
                lib = 'libhip_hcc.so'
            self._runtime = ctypes.cdll.LoadLibrary(lib)
        return self._runtime

    def runtime_call(self, funcname, argtypes, args):
        funcname = self.backend + funcname
        func = getattr(self.runtime, funcname)
        func.argtypes = argtypes
        if func(*args) != 0:
            raise RuntimeError(f'GPU runtime function {funcname} failed')

    def alloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self.runtime_call('Malloc', [ctypes.c_void_p, ctypes.c_size_t],
                          [ctypes.byref(ptr), nbytes])
        return ptr.value

    def free(self, ptr, nbytes):
        self.runtime_call('Free', [ctypes.c_void_p], [ptr])

    def memcpy(self, dst, src, nbytes, kind='Default'):
        kind = {
            'HostToHost': 0,
            'HostToDevice': 1,
            'DeviceToHost': 2,
            'DeviceToDevice': 3,
            'Default': 4
        }[kind]
        self.runtime_call(
            'Memcpy',
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
            [dst, src, nbytes, kind])


class BasicStencilMixin(StencilMixin):
    loop = benchmark.Parameter('loop kind', '1D', choices=['1D', '3D'])
    block_size = benchmark.Parameter('block_size', (1024, 1, 1))

    def setup(self):
        super().setup()
        if self.loop == '1D' and sum(b != 1 for b in self.block_size) == 1:
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
                    backend=self.backend)


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
                    backend=self.backend)
