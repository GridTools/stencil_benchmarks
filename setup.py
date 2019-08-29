import glob
import os
import re
import setuptools


def read_file(*path):
    """Read file content."""
    package_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(package_path, *path), 'r') as text_file:
        return text_file.read()


def long_description():
    """Read long description from README.md."""
    return read_file('README.md')


def version():
    """Parse version info."""
    initfile_content = read_file('stencil_benchmarks', '__init__.py')
    match = re.search(r"__version__ = [']([^']*)[']", initfile_content, re.M)
    if match:
        return match.group(1)
    raise RuntimeError('Unable to find version string')


setuptools.setup(
    name='stencil_benchmarks',
    version=version(),
    author='Felix Thaler',
    author_email='thaler@cscs.ch',
    description='Stencil code benchmarks.',
    long_description=long_description(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    scripts=glob.glob('bin/sbench*'),
    install_requires=['click', 'numba', 'numpy', 'pandas', 'jinja2'])
