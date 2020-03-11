import abc
import ctypes
import os

from stencil_benchmarks.benchmark import Benchmark, ExecutionError, Parameter
from stencil_benchmarks.tools import cpphelpers, compilation, template


class StencilMixin(Benchmark):
    compiler = Parameter('compiler path', 'g++')
    compiler_flags = Parameter('compiler flags', '')
    platform_preset = Parameter('preset flags for specific hardware platform',
                                'native',
                                choices=['none', 'native'])
    print_code = Parameter('print generated code', False)
    streaming_stores = Parameter(
        'enable streaming/non-temporal store instructions '
        '(only supported by some implementations)', False)

    def setup(self):
        super().setup()

        template_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'templates',
            self.template_file())
        code = template.render(template_file, **self.template_args())

        if self.print_code:
            print(cpphelpers.format_code(code))

        if self.compiler.endswith('icpc'):
            os.environ['KMP_INIT_AT_FORK'] = '0'

        self.compiled = compilation.GnuLibrary(code, self.compile_command())

    def compile_command(self):
        command = [self.compiler]
        if self.platform_preset != 'none':
            command += ['-std=c++11', '-Wall', '-DNDEBUG']
            if self.compiler.endswith('icpc'):
                command += ['-qopenmp', '-ffreestanding', '-O3']
            else:
                command += ['-fopenmp', '-Ofast']

            if self.platform_preset == 'native':
                command += ['-march=native', '-mtune=native', '-Ofast']
        if self.compiler_flags:
            command += self.compiler_flags.split()
        return command

    @abc.abstractmethod
    def template_file(self):
        pass

    def template_args(self):
        return dict(args=self.args,
                    ctype=compilation.dtype_cname(self.dtype),
                    strides=self.strides,
                    domain=self.domain,
                    halo=self.halo,
                    alignment=self.alignment,
                    streaming_stores=self.streaming_stores)

    def run_stencil(self, data):
        offset = (self.halo, ) * 3
        time = ctypes.c_double()
        try:
            self.compiled.kernel(
                ctypes.byref(time),
                *(compilation.data_ptr(array, offset) for array in data))
        except compilation.ExecutionError as error:
            raise ExecutionError() from error
        return time.value
