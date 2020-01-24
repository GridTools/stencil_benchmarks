import ctypes

import numpy as np

from .alloc import malloc

_offset = 64


def alloc_array(shape,
                dtype,
                layout,
                alignment=0,
                index_to_align=None,
                alloc=None,
                apply_offset=False):
    shape = tuple(shape)
    ndim = len(shape)
    dtype = np.dtype(dtype)
    layout = tuple(layout)
    alignment = int(alignment)
    index_to_align = (0, ) * ndim if index_to_align is None else index_to_align
    alloc = malloc if alloc is None else alloc

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
    buffer = alloc(strides_product + alignment + _offset)
    if alignment:
        pointer_to_align = ctypes.addressof(
            ctypes.c_char.from_buffer(buffer)) + np.sum(
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
