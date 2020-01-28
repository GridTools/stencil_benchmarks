import unittest

from stencil_benchmarks import benchmark

# pylint: disable=unexpected-keyword-arg,no-member


class SomeBenchmark(benchmark.Benchmark):
    parameter1 = benchmark.Parameter('some parameter', False)
    parameter2 = benchmark.Parameter('some other parameter', 1)

    def setup(self):
        self.somevalue = self.parameter2 // 2

    def run(self):
        return self.somevalue


class DerivedBenchmark(SomeBenchmark):
    parameter3 = benchmark.Parameter('third parameter', '0')

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
        expected = {
            'parameter1': benchmark.Parameter('some parameter', False),
            'parameter2': benchmark.Parameter('some other parameter', 1)
        }
        self.assertEqual(SomeBenchmark.parameters, expected)

    def test_derived_params(self):
        expected = {
            'parameter1': benchmark.Parameter('some parameter', False, bool),
            'parameter2': benchmark.Parameter('some other parameter', 1, int),
            'parameter3': benchmark.Parameter('third parameter', '0', str)
        }
        self.assertEqual(DerivedBenchmark.parameters, expected)

    def test_init(self):
        bmark = SomeBenchmark(parameter1=True, parameter2=42)
        self.assertEqual(bmark.parameter1, True)
        self.assertEqual(bmark.parameter2, 42)
        self.assertEqual(bmark.somevalue, 21)
        self.assertEqual(bmark.parameters, {
            'parameter1': True,
            'parameter2': 42
        })
        with self.assertRaises(benchmark.ParameterError):
            SomeBenchmark(parameter1=3, parameter2='42')

    def test_derived_init(self):
        bmark = DerivedBenchmark(parameter1=True,
                                 parameter2=42,
                                 parameter3='5')
        self.assertEqual(bmark.parameter1, True)
        self.assertEqual(bmark.parameter2, 42)
        self.assertEqual(bmark.parameter3, '5')
        self.assertEqual(bmark.parameters, {
            'parameter1': True,
            'parameter2': 42,
            'parameter3': '5'
        })
        self.assertEqual(bmark.somevalue, 21)
        self.assertEqual(bmark.paramsum, 48)

    def test_run(self):
        bmark = SomeBenchmark(parameter1=False, parameter2=42)
        self.assertEqual(bmark.run(), 21)

    def test_reassign(self):
        bmark = DerivedBenchmark(parameter1=True,
                                 parameter2=42,
                                 parameter3='5')
        bmark.parameter1 = False
        self.assertFalse(bmark.parameter1)
        self.assertFalse(bmark.parameters['parameter1'])
        with self.assertRaises(benchmark.ParameterError):
            bmark.parameter1 = 3
