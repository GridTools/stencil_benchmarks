import os
import re

from ...benchmark import Benchmark, Parameter, ExecutionError
from ...tools import compilation, cpphelpers, template


class Native(Benchmark):
    array_size = Parameter('number of elements in arrays', 10000000)
    ntimes = Parameter('number of runs', 10)
    block_size = Parameter('threads per block', 1024)
    dtype = Parameter('data type in NumPy format, e.g. float32 or float64',
                      'float64')
    compiler = Parameter('compiler path', 'nvcc')
    compiler_flags = Parameter('compiler flags', '')
    axis = Parameter('compute grid dimension to use',
                     'x',
                     choices=['x', 'y', 'z'])
    vector_size = Parameter('vector size', 1)
    print_code = Parameter('print code', False)
    verify = Parameter('verify results', True)

    def setup(self):
        super().setup()

        template_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'cuda_hip.j2')
        code = template.render(template_file, **self.template_args())
        if self.print_code:
            print(cpphelpers.format_code(code))
        self.compiled = compilation.GnuLibrary(code,
                                               self.compile_command(),
                                               extension='.cu')

    def compile_command(self):
        command = [self.compiler]
        if self.compiler_flags:
            command += self.compiler_flags.split()
        return command

    def template_args(self):
        return dict(array_size=self.array_size,
                    axis=self.axis,
                    block_size=self.block_size,
                    ctype=compilation.dtype_cname(self.dtype),
                    ntimes=self.ntimes,
                    vector_size=self.vector_size,
                    verify=self.verify)

    def run(self):
        try:
            output = self.compiled.run()
        except compilation.ExecutionError as error:
            raise ExecutionError(*error.args) from error

        regex = re.compile(r'(Copy|Scale|Add|Triad): +'
                           r'([0-9.]+) +([0-9.]+) +'
                           r'([0-9.]+) +([0-9.]+)')
        results = []
        for match in regex.finditer(output):
            results.append({
                'name': match.group(1).lower(),
                'bandwidth': float(match.group(2)),
                'avg-time': float(match.group(3)),
                'time': float(match.group(4)),
                'max-time': float(match.group(5))
            })

        return results
