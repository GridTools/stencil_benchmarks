# Stencil Benchmarks
#
# Copyright (c) 2017-2021, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# SPDX-License-Identifier: BSD-3-Clause
import unittest

from stencil_benchmarks import benchmark

# pylint: disable=unexpected-keyword-arg,no-member


class SomeBenchmark(benchmark.Benchmark):
    parameter1 = benchmark.Parameter("some parameter", False)
    parameter2 = benchmark.Parameter("some other parameter", 1)

    def setup(self):
        self.somevalue = self.parameter2 // 2

    def run(self):
        return self.somevalue


class DerivedBenchmark(SomeBenchmark):
    parameter3 = benchmark.Parameter("third parameter", "0")

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
            "parameter1": benchmark.Parameter("some parameter", False),
            "parameter2": benchmark.Parameter("some other parameter", 1),
        }
        self.assertEqual(SomeBenchmark.parameters, expected)

    def test_derived_params(self):
        expected = {
            "parameter1": benchmark.Parameter("some parameter", False, bool),
            "parameter2": benchmark.Parameter("some other parameter", 1, int),
            "parameter3": benchmark.Parameter("third parameter", "0", str),
        }
        self.assertEqual(DerivedBenchmark.parameters, expected)

    def test_init(self):
        bmark = SomeBenchmark(parameter1=True, parameter2=42)
        self.assertEqual(bmark.parameter1, True)
        self.assertEqual(bmark.parameter2, 42)
        self.assertEqual(bmark.somevalue, 21)
        self.assertEqual(bmark.parameters, {"parameter1": True, "parameter2": 42})
        with self.assertRaises(benchmark.ParameterError):
            SomeBenchmark(parameter1=3, parameter2="42")

    def test_derived_init(self):
        bmark = DerivedBenchmark(parameter1=True, parameter2=42, parameter3="5")
        self.assertEqual(bmark.parameter1, True)
        self.assertEqual(bmark.parameter2, 42)
        self.assertEqual(bmark.parameter3, "5")
        self.assertEqual(
            bmark.parameters, {"parameter1": True, "parameter2": 42, "parameter3": "5"}
        )
        self.assertEqual(bmark.somevalue, 21)
        self.assertEqual(bmark.paramsum, 48)

    def test_run(self):
        bmark = SomeBenchmark(parameter1=False, parameter2=42)
        self.assertEqual(bmark.run(), 21)

    def test_reassign(self):
        bmark = DerivedBenchmark(parameter1=True, parameter2=42, parameter3="5")
        bmark.parameter1 = False
        self.assertFalse(bmark.parameter1)
        self.assertFalse(bmark.parameters["parameter1"])
        with self.assertRaises(benchmark.ParameterError):
            bmark.parameter1 = 3
