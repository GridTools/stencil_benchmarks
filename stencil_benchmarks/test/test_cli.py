import unittest

from stencil_benchmarks import cli


class TestMain(unittest.TestCase):
    def test_basic(self):
        main = cli.main()
        main(args=[
            '--executions', '10', 'stencils', 'numpy', 'copy', '--runs', '10',
            '--domain', '10', '10', '10'
        ],
             standalone_mode=False)
