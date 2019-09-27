import unittest
import warnings

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
                     '   5 }')
        try:
            self.assertEqual(cpphelpers.format_code(code), formatted)
        except FileNotFoundError:
            warnings.warn(f'clang-format was not found')
