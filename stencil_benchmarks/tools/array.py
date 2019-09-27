import ctypes
import weakref

import numpy as np


def alloc_buffer(nbytes, alloc, free):
    pointer = alloc(int(nbytes))
    buffer = (ctypes.c_byte * nbytes).from_address(pointer)
    weakref.finalize(buffer, free, pointer, nbytes)
    return buffer


_LIBC = ctypes.cdll.LoadLibrary('libc.so.6')
_LIBC.malloc.restype = ctypes.c_void_p
_LIBC.malloc.argtypes = [ctypes.c_size_t]
_LIBC.free.argtypes = [ctypes.c_void_p]

_LIBC.mmap.restype = ctypes.c_void_p
_LIBC.mmap.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t
]
_LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def cmalloc(nbytes):
    pointer = _LIBC.malloc(nbytes)
    if not pointer:
        raise RuntimeError('could not allocate memory')
    return pointer


def cfree(pointer, nbytes):
    _LIBC.free(pointer)


def huge_alloc(nbytes):
    pointer = _LIBC.mmap(0, nbytes, 3, 278562, -1, 0)
    if not pointer:
        raise RuntimeError('could not allocate memory')
    return pointer


def huge_free(pointer, nbytes):
    _LIBC.munmap(pointer, nbytes)


_offset = 64


def alloc_array(shape,
                dtype,
                layout,
                alignment=0,
                index_to_align=None,
                alloc=None,
                free=None,
                apply_offset=False):
    shape = tuple(shape)
    ndim = len(shape)
    dtype = np.dtype(dtype)
    layout = tuple(layout)
    alignment = int(alignment)
    index_to_align = (0, ) * ndim if index_to_align is None else index_to_align
    alloc = cmalloc if alloc is None else alloc
    free = cfree if free is None else free

    if tuple(sorted(layout)) != tuple(range(ndim)):
        raise ValueError('invalid layout specification')
    if alignment < 0:
        raise ValueError('alignment must be non-negative')
    if not ndim == len(shape) == len(layout) == len(index_to_align):
        raise ValueError('dimension mismatch')

    strides = [0 for _ in range(ndim)]

    strides_product = dtype.itemsize
    for layout_value in range(ndim - 1, -1, -1):
        dimension = layout.index(layout_value)
        strides[dimension] = strides_product
        strides_product *= shape[dimension]
        if layout_value == ndim - 1 and alignment:
            strides_product = (strides_product + alignment -
                               1) // alignment * alignment

    global _offset
    buffer = alloc_buffer(strides_product + alignment + _offset, alloc, free)
    if alignment:
        pointer_to_align = ctypes.addressof(buffer) + np.sum(
            np.array(strides) * np.array(index_to_align))
        aligned_pointer = (pointer_to_align + alignment -
                           1) // alignment * alignment
        offset = aligned_pointer - pointer_to_align
    else:
        offset = 0
    if apply_offset:
        offset += _offset
        _offset *= 2
        if _offset > 2048:
            _offset = 64
    return np.ndarray(shape=shape,
                      dtype=dtype,
                      buffer=buffer,
                      offset=offset,
                      strides=strides)


def nbytes(data):
    last_index = np.sum((np.array(data.shape) - 1) * np.array(data.strides))
    return last_index + data.itemsize
