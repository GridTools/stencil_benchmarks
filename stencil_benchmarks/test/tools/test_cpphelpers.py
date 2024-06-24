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
import subprocess
import unittest

from stencil_benchmarks.tools import cpphelpers


class TestFormatCode(unittest.TestCase):
    def test_hello_world(self):
        code = (
            "#include <iostream>\n"
            "int main(int argc ,const char**argv){"
            'std::cout<<"Hello World!"<<std::endl;'
            "return 0;}"
        )
        formatted = (
            "   1 #include <iostream>\n"
            "   2 int main(int argc, const char **argv) {\n"
            '   3   std::cout << "Hello World!" << std::endl;\n'
            "   4   return 0;\n"
            "   5 }\n"
        )

        try:
            subprocess.run(["clang-format", "--version"], check=True)
            clang_format_available = True
        except FileNotFoundError:
            clang_format_available = False

        if clang_format_available:
            self.assertEqual(cpphelpers.format_code(code), formatted)
        else:
            with self.assertWarnsRegex(
                Warning, "code not formatted: could not find clang-format"
            ):
                cpphelpers.format_code(code)
