# Stencil Benchmarks
#
# Copyright (c) 2017-2020, ETH Zurich and MeteoSwiss
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
    package_data={'': ['*.j2', '*.pyi']},
    ext_modules=ext_modules,
    entry_points={
        'console_scripts': [
            'sbench=stencil_benchmarks.scripts.sbench:main',
            'sbench-analyze=stencil_benchmarks.scripts.sbench_analyze:main',
            'sbench-a100-collection=stencil_benchmarks.scripts'
            '.sbench_a100_collection:main',
            'sbench-v100-collection=stencil_benchmarks.scripts'
            '.sbench_v100_collection:main',
            'sbench-mi50-collection=stencil_benchmarks.scripts'
            '.sbench_mi50_collection:main',
            'sbench-mi100-collection=stencil_benchmarks.scripts'
            '.sbench_mi100_collection:main',
            'sbench-a64fx-collection=stencil_benchmarks.scripts'
            '.sbench_a64fx_collection:main',
            'sbench-rome-collection=stencil_benchmarks.scripts'
            '.sbench_rome_collection:main'
        ]
    },
    install_requires=[
        'click', 'numpy', 'pandas', 'jinja2', 'matplotlib', 'pybind11'
    ],
    extras_require={
        'gt4py-dace':
        ['gt4py[dace] @ git+https://github.com/gridtools/gt4py.git@dace']
    },
    setup_requires=['pybind11'],
    zip_safe=False)
