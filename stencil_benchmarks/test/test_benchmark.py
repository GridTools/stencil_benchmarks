import unittest

from stencil_benchmarks import benchmark

# pylint: disable=unexpected-keyword-arg,no-member


class SomeBenchmark(benchmark.Benchmark):
    parameter1 = benchmark.Parameter('some parameter', bool)
    parameter2 = benchmark.Parameter('some other parameter', int)

    def setup(self):
        self.somevalue = self.parameter2 // 2

    def run(self):
        return self.somevalue


class DerivedBenchmark(SomeBenchmark):
    parameter3 = benchmark.Parameter('third parameter', str)

    def setup(self):
        super().setup()
        self.paramsum = sum(int(param) for param in self.parameters.values())

    def run(self):
        return self.paramsum


class BenchmarkTest(unittest.TestCase):
    def test_registration(self):
        self.assertIn(SomeBenchmark, benchmark.REGISTRY)
        self.assertIn(DerivedBenchmark, benchmark.REGISTRY)

    def test_params(self):
        self.assertEqual(
            SomeBenchmark.parameters, {
                'parameter1': benchmark.Parameter('some parameter', bool),
                'parameter2': benchmark.Parameter('some other parameter', int)
            })

    def test_derived_params(self):
        self.assertEqual(
            DerivedBenchmark.parameters, {
                'parameter1': benchmark.Parameter('some parameter', bool),
                'parameter2': benchmark.Parameter('some other parameter', int),
                'parameter3': benchmark.Parameter('third parameter', str)
            })

    def test_init(self):
        bmark = SomeBenchmark(parameter1=True, parameter2=42)
        self.assertEqual(bmark.parameter1, True)
        self.assertEqual(bmark.parameter2, 42)
        self.assertEqual(bmark.somevalue, 21)
        self.assertEqual(bmark.parameters, {
            'parameter1': True,
            'parameter2': 42
        })

    def test_derived_init(self):
        bmark = DerivedBenchmark(parameter1=True,
                                 parameter2=42,
                                 parameter3='5')
        self.assertEqual(bmark.parameter1, True)
        self.assertEqual(bmark.parameter2, 42)
        self.assertEqual(bmark.parameter3, '5')
        self.assertEqual(bmark.somevalue, 21)
        self.assertEqual(bmark.paramsum, 48)

    def test_run(self):
        bmark = SomeBenchmark(parameter1=False, parameter2=42)
        self.assertEqual(bmark.run(), 21)
