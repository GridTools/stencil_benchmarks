import unittest

from stencil_benchmarks import cli


class TestMain(unittest.TestCase):
    def test_basic(self):
        main = cli.main()
        main(args=[
            '--executions', '10', 'stencils', 'copy', 'numpy', 'numpy',
            '--runs', '10', '--domain', '1000', '1000', '80'
        ],
             standalone_mode=False)
