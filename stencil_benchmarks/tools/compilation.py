import ctypes
import subprocess
import tempfile
from typing import Any, List, Optional, Tuple, Union

import numpy as np


class CompilationError(RuntimeError):
    pass


def gnu_library(compile_command: List[str],
                code: str,
                extension: Optional[str] = None,
                libraries: Optional[List[str]] = None):
    if compile_command[0].endswith('nvcc'):
        lib_flags = ['-Xcompiler', '-shared', '-Xcompiler', '-fPIC']
    else:
        lib_flags = ['-shared', '-fPIC']
    if extension is None:
        extension = '.cpp'
    if libraries is None:
        libraries = []
    with tempfile.NamedTemporaryFile(suffix=extension) as srcfile:
        srcfile.write(code.encode())
        srcfile.flush()

        with tempfile.NamedTemporaryFile(suffix='.so') as library:
            try:
                subprocess.run(compile_command + lib_flags +
                               ['-o', library.name, srcfile.name] + libraries,
                               check=True)
            except subprocess.CalledProcessError as error:
                raise CompilationError(*error.args) from None
            return ctypes.cdll.LoadLibrary(library.name)


def gnu_func(compile_command: List[str],
             code: str,
             funcname: str,
             restype: List[Any] = None,
             argtypes: List[Any] = None,
             source_extension: Optional[str] = None,
             libraries: Optional[List[str]] = None):
    func = getattr(
        gnu_library(compile_command, code, source_extension, libraries),
        funcname)
    if restype is not None:
        func.restype = dtype_as_ctype(restype)
    if argtypes is not None:
        func.argtypes = [dtype_as_ctype(t) for t in argtypes]
    return func


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


def data_ptr(array: np.ndarray,
             offset=Union[int, Tuple[int, ...], None]) -> ctypes.c_void_p:
    if offset is None:
        offset = 0
    elif isinstance(offset, int):
        offset *= array.dtype.itemsize
    else:
        offset = int(np.sum(np.asarray(array.strides) * np.asarray(offset)))
    return ctypes.cast(array.ctypes.data + offset, ctypes.c_void_p)
