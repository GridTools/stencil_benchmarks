import unittest
import warnings

from stencil_benchmarks import benchmark


class TestAllBenchmarks(unittest.TestCase):
    def test_all(self):
        for bmark in benchmark.REGISTRY:
            bmark_name = bmark.__module__ + '.' + bmark.__name__
            with self.subTest(benchmark=bmark_name):
                try:
                    bmark_instance = bmark()
                except benchmark.ParameterError:
                    warnings.warn(f'{bmark_name} was not executed due '
                                  f'to missing default parameters')
                    continue
                bmark_instance.run()
