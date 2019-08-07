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


class TestLoop(unittest.TestCase):
    def test_forward(self):
        expected = cpphelpers.format_code(
            'for (int i = 3; i < 10; i += 1) {a = b;}')
        self.assertEqual(cpphelpers.loop('i', range(3, 10), 'a = b;'),
                         expected)

    def test_backward(self):
        expected = cpphelpers.format_code(
            'for (int i = 9; i > 2; i += -1) {foo(i);}')
        self.assertEqual(cpphelpers.loop('i', range(9, 2, -1), 'foo(i);'),
                         expected)


class TestNestedLoop(unittest.TestCase):
    def test_1d(self):
        expected = cpphelpers.format_code(
            'for (int i = 3; i < 10; i += 1) {foo(i);}')
        self.assertEqual(
            cpphelpers.nested_loops(['i'], [range(3, 10)], 'foo(i);'),
            expected)

    def test_2d(self):
        expected = cpphelpers.format_code(
            'for (int i = 3; i < 10; i += 1) {for (int j = 42; j > -1; j += -1) {foo(i, j);}}'
        )
        self.assertEqual(
            cpphelpers.nested_loops(
                ['i', 'j'], [range(3, 10), range(42, -1, -1)], 'foo(i, j);'),
            expected)
