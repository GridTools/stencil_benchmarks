import unittest

from stencil_benchmarks import benchmark


class TestAllBenchmarks(unittest.TestCase):
    def test_all(self):
        for bmark in benchmark.REGISTRY:
            with self.subTest(benchmark=bmark.__module__ + '.' +
                              bmark.__name__):
                bmark().run()
