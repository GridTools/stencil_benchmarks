import subprocess
import unittest

from stencil_benchmarks.tools import cpphelpers


class TestFormatCode(unittest.TestCase):
    def test_hello_world(self):
        code = ('#include <iostream>\n'
                'int main(int argc ,const char**argv){'
                'std::cout<<"Hello World!"<<std::endl;'
                'return 0;}')
        formatted = ('   1 #include <iostream>\n'
                     '   2 int main(int argc, const char **argv) {\n'
                     '   3   std::cout << "Hello World!" << std::endl;\n'
                     '   4   return 0;\n'
                     '   5 }\n')

        try:
            subprocess.run(['clang-format', '--version'], check=True)
            clang_format_available = True
        except FileNotFoundError:
            clang_format_available = False

        if clang_format_available:
            self.assertEqual(cpphelpers.format_code(code), formatted)
        else:
            with self.assertWarnsRegex(
                    Warning,
                    'code not formatted: could not find clang-format'):
                cpphelpers.format_code(code)
