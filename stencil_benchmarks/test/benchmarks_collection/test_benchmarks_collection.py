import unittest

from stencil_benchmarks import benchmark


class TestAllBenchmarks(unittest.TestCase):
    def test_all(self):
        for bmark in benchmark.REGISTRY:
            bmark().run()
