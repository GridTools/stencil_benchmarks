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


class pybind11_include:
    def __init__(self, user):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


def pybind11_extension(m):
    return setuptools.Extension(
        m, [m.replace('.', '/') + '.cpp'],
        include_dirs=[pybind11_include(False),
                      pybind11_include(True)],
        language='c++')


ext_modules = [
    pybind11_extension('stencil_benchmarks.tools.alloc'),
    pybind11_extension('stencil_benchmarks.tools.parallel')
]

setuptools.setup(
    name='stencil_benchmarks',
    version=version(),
    author='Felix Thaler',
    author_email='thaler@cscs.ch',
    description='Stencil code benchmarks.',
    long_description=long_description(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    package_data={'': ['*.j2']},
    ext_modules=ext_modules,
    entry_points={
        'console_scripts': [
            'sbench=stencil_benchmarks.scripts.sbench:main',
            'sbench-analyze=stencil_benchmarks.scripts.sbench_analyze:main',
            'sbench-cudahip-collection=stencil_benchmarks.scripts'
            '.sbench_cudahip_collection:main',
            'sbench-openmp-collection=stencil_benchmarks.scripts'
            '.sbench_openmp_collection:main'
        ]
    },
    install_requires=[
        'click', 'numpy', 'pandas', 'jinja2', 'matplotlib', 'pybind11'
    ],
    setup_requires=['pybind11'])
