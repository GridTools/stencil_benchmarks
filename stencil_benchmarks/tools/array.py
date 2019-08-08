import ctypes
import weakref

import numpy as np


def alloc_buffer(nbytes, alloc, free):
    pointer = alloc(int(nbytes))
    buffer = (ctypes.c_byte * nbytes).from_address(pointer)
    weakref.finalize(buffer, free, pointer)
    return buffer


_LIBC = ctypes.cdll.LoadLibrary('libc.so.6')


def cmalloc(nbytes):
    return _LIBC.malloc(int(nbytes))


def cfree(pointer):
    _LIBC.free(int(pointer))


def alloc_array(shape,
                dtype,
                layout,
                alignment=0,
                index_to_align=None,
                alloc=None,
                free=None):
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

    buffer = alloc_buffer(strides_product + alignment, alloc, free)
    if alignment:
        pointer_to_align = ctypes.addressof(buffer) + np.sum(
            np.array(strides) * np.array(index_to_align))
        aligned_pointer = (pointer_to_align + alignment -
                           1) // alignment * alignment
        offset = aligned_pointer - pointer_to_align
    else:
        offset = 0
    return np.ndarray(shape=shape,
                      dtype=dtype,
                      buffer=buffer,
                      offset=offset,
                      strides=strides)
