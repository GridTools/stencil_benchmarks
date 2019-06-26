import unittest

from stencil_benchmarks import benchmark

# pylint: disable=unexpected-keyword-arg,no-member


class SomeBenchmark(benchmark.Benchmark):
    parameter1 = benchmark.Parameter('some parameter', bool)
    parameter2 = benchmark.Parameter('some other parameter', int)

    def __init__(self):
        self.someparameter = self.parameter2 // 2

    def run(self):
        return self.someparameter


class BenchmarkTest(unittest.TestCase):
    def test_registration(self):
        self.assertIn(SomeBenchmark, benchmark.REGISTRY)

    def test_init(self):
        bmark = SomeBenchmark(parameter1=True, parameter2=42)
        self.assertEqual(bmark.parameter1, True)
        self.assertEqual(bmark.parameter2, 42)
        self.assertEqual(bmark.someparameter, 21)

    def test_run(self):
        bmark = SomeBenchmark(parameter1=False, parameter2=42)
        self.assertEqual(bmark.run(), 21)

    def test_parameters(self):
        self.assertEqual(
            SomeBenchmark.parameters, {
                'parameter1': benchmark.Parameter('some parameter', bool),
                'parameter2': benchmark.Parameter('some other parameter', int)
            })
