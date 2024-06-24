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
import contextlib
import ctypes
import io
import os
import pathlib
import subprocess
import tempfile
from typing import (Any, Callable, Iterator, List, Optional, TextIO, Tuple,
                    Union)
import warnings

import numpy as np


class CompilationError(RuntimeError):
    pass


class ExecutionError(RuntimeError):
    pass


@contextlib.contextmanager
def _redirect_output(fileno: int, target: TextIO) -> Iterator[None]:
    backup = os.dup(fileno)

    with tempfile.TemporaryFile() as tmpfile:
        os.dup2(tmpfile.fileno(), fileno)

        yield

        tmpfile.seek(0)
        target.write(tmpfile.read().decode())

    os.dup2(backup, fileno)
    os.close(backup)


@contextlib.contextmanager
def _capture_output(stdout: TextIO, stderr: TextIO) -> Iterator[None]:
    """Context manager to capture the output written to stdout and stderr.

    Works by temporarily redefining the stdout and stderr file descriptors.
    Thus, in contrast to contextlib.redirect_stdout() and
    contextlib.redirect_stderr(), this also works for (C/C++) library calls.

    Parameters
    ----------
    stdout : TextIO
        TextIO to capture stdout.
    stderr : TextIO
        TextIO to capture stderr.
    """
    with _redirect_output(1, stdout):
        with _redirect_output(2, stderr):
            yield


class GnuLibrary:
    def __init__(self,
                 code: str,
                 compile_command: Optional[List[str]] = None,
                 extension: Optional[str] = None):
        """Compile and load a C/C++-library.

        Parameters
        ----------
        code : str
            Code to compile.
        compile_command : list of str
            Command to use for compilation.
        extension : str
            File extension to use for code file.
        Examples
        --------
        >>> lib = GnuLibrary('''
        ...                  #include <stdio.h>
        ...                  int useless_function() {
        ...                      printf("Hello world!");
        ...                      fflush(stdout);
        ...                      return 0;
        ...                  }
        ...                  ''',
        ...                  extension='.c')
        >>> lib.useless_function()
        'Hello world!'
        """
        if extension is None:
            extension = '.cpp'
        if compile_command is None:
            compile_command = ['gcc'] if extension.lower() == '.c' else ['g++']

        if compile_command[0].endswith('nvcc'):
            compile_command += ['-Xcompiler', '-shared', '-Xcompiler', '-fPIC']
        else:
            compile_command += ['-shared', '-fPIC']

        output_dir = pathlib.Path("benchmarks_source_code")
        output_dir.mkdir(exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=extension, dir=output_dir, delete=False) as srcfile:
            srcfile.write(code.encode())
            srcfile.flush()

            with tempfile.NamedTemporaryFile(suffix='.so') as library:
                result = subprocess.run(
                    [compile_command[0], '-o', library.name, srcfile.name] +
                    compile_command[1:],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
                if result.returncode != 0:
                    raise CompilationError(result.stderr.decode())
                if result.stdout or result.stderr:
                    warnings.warn('unexpected compilation output: ' +
                                  result.stdout.decode() +
                                  result.stderr.decode())
                self._library = ctypes.cdll.LoadLibrary(library.name)

    def __getattr__(self, attr: str) -> Callable[[Any], str]:
        """Access library function by name.

        The returned callable is a wrapper around the C/C++ library function.
        The wrapped C/C++ function must return zero on success and a non-zero
        int in case of failure. Arbitrary arguments can be passed when wrapped
        into corresponding `ctypes` objects, optionally the callable takes a
        keyword argument `argtypes` to define the input parameter types.

        Parameter
        ---------
        attr : str
            Name of the C/C++ library function.

        Returns
        -------
        Callable
            Wrapper around the C/C++ library function.
        """
        func = getattr(self._library, attr)

        def wrapper(*args, argtypes=None):
            if argtypes is not None:
                func.argtypes = argtypes
            stdout = io.StringIO()
            stderr = io.StringIO()
            with _capture_output(stdout, stderr):
                result = func(*args)
            stdout = stdout.getvalue()
            stderr = stderr.getvalue()

            if result != 0:
                raise ExecutionError(stderr)
            elif stderr:
                warnings.warn(
                    f'unexpected output in call to {attr}(…) to stderr:\n' +
                    stderr)

            return stdout

        wrapper.__name__ = attr
        return wrapper


def dtype_as_ctype(dtype: np.dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == 'f':
        if dtype.itemsize == 4:
            return ctypes.c_float
        if dtype.itemsize == 8:
            return ctypes.c_double
    elif dtype.kind == 'i':
        if dtype.itemsize == 1:
            return ctypes.c_int8
        if dtype.itemsize == 2:
            return ctypes.c_int16
        if dtype.itemsize == 4:
            return ctypes.c_int32
        if dtype.itemsize == 8:
            return ctypes.c_int64
    elif dtype.kind == 'u':
        if dtype.itemsize == 1:
            return ctypes.c_uint8
        if dtype.itemsize == 2:
            return ctypes.c_uint16
        if dtype.itemsize == 4:
            return ctypes.c_uint32
        if dtype.itemsize == 8:
            return ctypes.c_uint64
    raise NotImplementedError(f'Conversion of type {dtype} is not supported')


def ctype_cname(ctype) -> str:
    if ctype is ctypes.c_float:
        return 'float'
    if ctype is ctypes.c_double:
        return 'double'

    if ctype is ctypes.c_int8:
        return 'std::int8_t'
    if ctype is ctypes.c_int16:
        return 'std::int16_t'
    if ctype is ctypes.c_int32:
        return 'std::int32_t'
    if ctype is ctypes.c_int64:
        return 'std::int64_t'

    if ctype is ctypes.c_uint8:
        return 'std::uint8_t'
    if ctype is ctypes.c_uint16:
        return 'std::uint16_t'
    if ctype is ctypes.c_uint32:
        return 'std::uint32_t'
    if ctype is ctypes.c_uint64:
        return 'std::uint64_t'

    raise NotImplementedError(f'Conversion of type {ctype} is not supported')


def dtype_cname(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype.kind == 'f':
        if dtype.itemsize == 2:
            return 'half'
        if dtype.itemsize == 4:
            return 'float'
        if dtype.itemsize == 8:
            return 'double'

    if dtype.kind == 'i':
        return f'std::int{dtype.itemsize * 8}_t'

    if dtype.kind == 'u':
        return f'std::uint{dtype.itemsize * 8}_t'

    raise NotImplementedError(f'Conversion of type {dtype} is not supported')


def data_ptr(
        array: np.ndarray,
        offset: Union[int, Tuple[int, ...], None] = None) -> ctypes.c_void_p:
    if offset is None:
        offset = 0
    elif isinstance(offset, int):
        offset *= array.dtype.itemsize
    else:
        offset = int(np.sum(np.asarray(array.strides) * np.asarray(offset)))
    return ctypes.cast(array.ctypes.data + offset, ctypes.c_void_p)
