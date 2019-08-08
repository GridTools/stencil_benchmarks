import unittest

from stencil_benchmarks.tools import cpphelpers


class TestFormatCode(unittest.TestCase):
    def test_hello_world(self):
        code = ('#include <iostream>\n'
                'int main(int argc ,const char**argv){'
                'std::cout<<"Hello World!"<<std::endl;'
                'return 0;}')
        formatted = ('#include <iostream>\n'
                     'int main(int argc, const char **argv) {\n'
                     '  std::cout << "Hello World!" << std::endl;\n'
                     '  return 0;\n'
                     '}')
        self.assertEqual(cpphelpers.format_code(code), formatted)