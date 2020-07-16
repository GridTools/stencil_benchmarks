import unittest
import warnings

from stencil_benchmarks import benchmark


class TestAllBenchmarks(unittest.TestCase):
    def test_all(self):
        untested = []
        for bmark in benchmark.REGISTRY:
            bmark_name = bmark.__module__ + '.' + bmark.__name__
            with self.subTest(benchmark=bmark_name):
                try:
                    bmark_instance = bmark()
                except benchmark.ParameterError as e:
                    warnings.warn(e)
                    untested.append(bmark_name)
                    continue
                bmark_instance.run()

        if untested:
            warnings.warn('The following benchmarks were not tested:\n' +
                          '\n'.join(untested))
