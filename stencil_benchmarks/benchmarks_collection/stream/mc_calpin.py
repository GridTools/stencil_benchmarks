import os
import subprocess
import re
import weakref

import numpy as np

from ...benchmark import Benchmark, Parameter, ParameterError, ExecutionError
from ...tools import compilation, template


class Original(Benchmark):
    array_size = Parameter('number of elements in arrays', 10000000)
    ntimes = Parameter('number of runs', 10)
    offset = Parameter('allocation offset', 0)
    dtype = Parameter('data type in NumPy format, e.g. float32 or float64',
                      'float64')
    compiler = Parameter('compiler path', 'gcc')
    compiler_flags = Parameter('compiler flags', '')
    platform_preset = Parameter('preset flags for specific hardware platform',
                                'basic',
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
                re.search(r'^void tuned_STREAM_Copy().*', code,
                          re.MULTILINE | re.DOTALL).group())

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
                    'bandwidth': match.group(2),
                    'avg-time': match.group(3),
                    'time': match.group(4),
                    'max-time': match.group(5)
                })

        return results


class Native(Original):
    architecture = Parameter('hardware architecture',
                             'x86-avx',
                             choices=['x86-sse', 'x86-avx', 'x86-avx512'])
    unroll_factor = Parameter('loop unroll factor', 1)
    fma = Parameter('use fused multiply-add instructions', True)

    def template_file(self):
        return 'native.j2'

    def template_args(self):
        itemsize = np.dtype(self.dtype).itemsize
        if self.architecture.startswith('x86'):
            includes = ['immintrin.h']
            ops = dict(vector_load='load_p',
                       vector_store='stream_p',
                       vector_broadcast='set1_p',
                       vector_mul='mul_p',
                       vector_add='add_p',
                       vector_fmadd='fmadd_p')
            if self.architecture.endswith('sse'):
                vector_size = 16 // itemsize
                if self.dtype == 'float32':
                    vector_type = '__m128'
                    for op in ops:
                        ops[op] = f'_mm_{ops[op]}s'
                elif self.dtype == 'float64':
                    vector_type = '__m128d'
                    for op in ops:
                        ops[op] = f'_mm_{ops[op]}d'
                else:
                    raise ParameterError(f'dtype {self.dtype} not supported')
            elif self.architecture.endswith('avx'):
                vector_size = 32 // itemsize
                if self.dtype == 'float32':
                    vector_type = '__m256'
                    for op in ops:
                        ops[op] = f'_mm256_{ops[op]}s'
                elif self.dtype == 'float64':
                    vector_type = '__m256d'
                    for op in ops:
                        ops[op] = f'_mm256_{ops[op]}d'
                else:
                    raise ParameterError(f'dtype {self.dtype} not supported')
            elif self.architecture.endswith('avx512'):
                vector_size = 64 // itemsize
                if self.dtype == 'float32':
                    vector_type = '__m512'
                    for op in ops:
                        ops[op] = f'_mm512_{ops[op]}s'
                elif self.dtype == 'float64':
                    vector_type = '__m512d'
                    for op in ops:
                        ops[op] = f'_mm512_{ops[op]}d'
                else:
                    raise ParameterError(f'dtype {self.dtype} not supported')
        else:
            raise ParameterError(
                f'unsupported architecture {self.architecture}')
        return dict(**super().template_args(),
                    unroll_factor=self.unroll_factor,
                    fma=self.fma,
                    includes=includes,
                    vector_size=vector_size,
                    vector_type=vector_type,
                    **ops)
