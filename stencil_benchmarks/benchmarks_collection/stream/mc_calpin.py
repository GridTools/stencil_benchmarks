import os
import subprocess
import re
import weakref

import numpy as np

from ...benchmark import Benchmark, Parameter, ParameterError, ExecutionError
from ...tools import compilation, cpphelpers, template


class Original(Benchmark):
    array_size = Parameter('number of elements in arrays', 10000000)
    ntimes = Parameter('number of runs', 10)
    offset = Parameter('allocation offset', 0)
    dtype = Parameter('data type in NumPy format, e.g. float32 or float64',
                      'float64')
    compiler = Parameter('compiler path', 'gcc')
    compiler_flags = Parameter('compiler flags', '')
    platform_preset = Parameter('preset flags for specific hardware platform',
                                'native',
                                choices=['none', 'basic', 'native'])
    print_kernels = Parameter('print kernel code', False)

    def setup(self):
        super().setup()

        if self.compiler.endswith('icpc'):
            os.environ['KMP_INIT_AT_FORK'] = '0'

        template_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.template_file())
        code = template.render(template_file, **self.template_args())
        if self.print_kernels:
            print(
                re.search(r'/\* Tuned kernels. \*/(.*)',
                          cpphelpers.format_code(code),
                          re.MULTILINE | re.DOTALL).group(1))

        self.executable = compilation.gnu_executable(self.compile_command(),
                                                     code,
                                                     extension='.c')
        weakref.finalize(self, os.remove, self.executable)

    def compile_command(self):
        command = [self.compiler]
        if self.platform_preset != 'none':
            command += ['-Wall', '-DNDEBUG']
            if self.compiler.endswith('icc'):
                command += [
                    '-qopenmp', '-ffreestanding', '-O3', '-mcmodel=large',
                    '-qopt-streaming-stores=always'
                ]
            else:
                command += [
                    '-fopenmp', '-ffreestanding', '-Ofast', '-mcmodel=large'
                ]
            if self.platform_preset == 'native':
                command += ['-march=native', '-mtune=native']
        if self.compiler_flags:
            command += self.compiler_flags.split()
        return command

    def template_file(self):
        return 'original.j2'

    def template_args(self):
        return dict(array_size=self.array_size,
                    ntimes=self.ntimes,
                    offset=self.offset,
                    ctype=compilation.dtype_cname(self.dtype))

    def run(self):
        run = subprocess.run([self.executable],
                             check=True,
                             stdout=subprocess.PIPE)
        output = run.stdout.decode()

        match = re.search(r'Failed Validation.*', output,
                          re.MULTILINE | re.DOTALL)
        if match:
            raise ExecutionError(match.group())

        regex = re.compile(r'(Copy|Scale|Add|Triad): +'
                           r'([0-9.]+) +([0-9.]+) +'
                           r'([0-9.]+) +([0-9.]+)')
        results = []
        for line in output.splitlines():
            match = regex.match(line)
            if match:
                results.append({
                    'name': match.group(1).lower(),
                    'bandwidth': float(match.group(2)),
                    'avg-time': float(match.group(3)),
                    'time': float(match.group(4)),
                    'max-time': float(match.group(5))
                })

        return results


class _GccVecOps:
    def __init__(self, vector_size, dtype):
        itemsize = np.dtype(dtype).itemsize
        self.preamble = (
            f'typedef {compilation.dtype_cname(dtype)} vec_t '
            f'__attribute__((vector_size({vector_size * itemsize})));')
        self.vector_size = vector_size
        self.vector_type = 'vec_t'
        self.includes = []

    def load(self, ptr):
        return f'*({self.vector_type}*){ptr}'

    def store(self, ptr, value):
        return f'*({self.vector_type}*){ptr} = {value}'

    def broadcast(self, value):
        return f'({self.vector_type}){{' + ', '.join(
            str(value) for _ in range(self.vector_size)) + '}'

    def mul(self, a, b):
        return f'({a} * {b})'

    def add(self, a, b):
        return f'({a} + {b})'

    def fma(self, a, b, c):
        return f'({a} * {b} + {c})'


class _x86Ops:
    def __init__(self, ext, dtype, streaming_stores):
        itemsize = np.dtype(dtype).itemsize
        if ext == 'sse':
            self.vector_size = 16 // itemsize
            self._prefix = '_mm'
            self.vector_type = '__m128'
            self.includes = ['xmmintrin.h', 'emmintrin.h']
        elif ext == 'avx':
            self.vector_size = 32 // itemsize
            self._prefix = '_mm256'
            self.vector_type = '__m256'
            self.includes = ['immintrin.h']
        elif ext == 'avx512':
            self.vector_size = 64 // itemsize
            self._prefix = '_mm512'
            self.vector_type = '__m512'
            self.includes = ['immintrin.h']
        else:
            raise ValueError('unsupported extension')
        if dtype == 'float32':
            self._suffix = 's'
        elif dtype == 'float64':
            self._suffix = 'd'
            self.vector_type += 'd'
        else:
            raise ValueError('unsupported dtype')
        self._streaming_stores = streaming_stores

    def load(self, ptr):
        return f'{self._prefix}_load_p{self._suffix}({ptr})'

    def store(self, ptr, value):
        if self._streaming_stores:
            return f'{self._prefix}_stream_p{self._suffix}({ptr}, {value})'
        return f'{self._prefix}_store_p{self._suffix}({ptr}, {value})'

    def broadcast(self, value):
        return f'{self._prefix}_set1_p{self._suffix}({value})'

    def mul(self, a, b):
        return f'{self._prefix}_mul_p{self._suffix}({a}, {b})'

    def add(self, a, b):
        return f'{self._prefix}_add_p{self._suffix}({a}, {b})'

    def fma(self, a, b, c):
        return f'{self._prefix}_fmadd_p{self._suffix}({a}, {b}, {c})'


class Native(Original):
    architecture = Parameter(
        'hardware architecture',
        'gcc-vec',
        choices=['gcc-vec', 'x86-sse', 'x86-avx', 'x86-avx512'])
    unroll_factor = Parameter('loop unroll factor', 1)
    fma = Parameter('use fused multiply-add instructions', True)
    vector_size = Parameter('vector size (only where applicable)', 0)
    streaming_stores = Parameter('bypass load of destination array', True)

    def setup(self):
        super().setup()

    def template_file(self):
        return 'native.j2'

    def template_args(self):
        if self.architecture == 'gcc-vec':
            ops = _GccVecOps(max(self.vector_size, 1), self.dtype)
        elif self.architecture.startswith('x86'):
            ops = _x86Ops(
                self.architecture.split('-')[1], self.dtype,
                self.streaming_stores)
        else:
            raise ParameterError(
                f'unsupported architecture {self.architecture}')
        return dict(**super().template_args(),
                    unroll_factor=self.unroll_factor,
                    ops=ops)
