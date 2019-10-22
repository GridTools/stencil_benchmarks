import ctypes
from typing import Callable, Dict, Optional, Tuple, Union
import weakref

import numpy as np


def alloc_buffer(nbytes: int, alloc: Callable[[int], int],
                 free: Callable[[int, int], None]):
    """Allocate a Python buffer object.

    Uses the given `alloc` and `free` functions to allocate,
    respectively free the data pointer.

    Calls `alloc(nbytes)` for allocation and `free(ptr, nbytes)`
    for deallocation.

    Parameters
    ----------
    nbytes : int
        Size of the buffer in bytes.
    alloc : func
        Memory allocation function. Must accept one parameter `nbytes`
        and return a pointer.
    free : func
        Memory freeing function. Must accept two parameters `ptr` and
        `nbytes`.

    Returns
    -------
    A Python memory byffer object.

    Examples
    --------
    >>> buffer = alloc_buffer(4, cmalloc, cfree)
    >>> view = memoryview(buffer).cast('B')
    >>> view.nbytes
    4
    >>> view[:] = bytes.fromhex('01020304')
    >>> view.tolist()
    [1, 2, 3, 4]
    """
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


def cmalloc(nbytes: int) -> int:
    """C malloc() wrapper.

    Parameters
    ----------
    nbytes : int
        Number of bytes to allocate.

    Returns
    -------
    int
        Pointer to a memory allocation of size `nbytes`.

    Examples
    --------
    >>> ptr = cmalloc(100)
    >>> print(f'ptr: 0x{ptr:02x}')
    ptr: 0x...
    >>> cfree(ptr)
    """
    pointer = _LIBC.malloc(nbytes)
    if not pointer:
        raise RuntimeError('could not allocate memory')
    return pointer


def cfree(pointer: int, nbytes: Optional[int] = None) -> None:
    """C free() wrapper.

    Parameters
    ----------
    pointer : int
        Pointer to memory which should be freed.
    nbytes : int, optional
        Number of bytes of the allocation (ignored).
        For compatibility with `alloc_buffer`.

    Examples
    --------
    >>> ptr = cmalloc(100)
    >>> cfree(ptr)
    """
    _LIBC.free(pointer)


def huge_alloc(nbytes: int) -> int:
    """Huge-page allocation with mmap.

    Parameters
    ----------
    nbytes : int
        Number of bytes to allocate.

    Returns
    -------
    int
        Pointer to a memory allocation of size `nbytes`.

    Examples
    --------
    >>> ptr = huge_alloc(100)
    >>> print(f'ptr: 0x{ptr:02x}')
    ptr: 0x...
    >>> huge_free(ptr, 100)
    """
    pointer = _LIBC.mmap(0, nbytes, 3, 278562, -1, 0)
    if not pointer:
        raise RuntimeError('could not allocate memory')
    return pointer


def huge_free(pointer: int, nbytes: int) -> None:
    """C free() wrapper.

    Parameters
    ----------
    pointer : int
        Pointer to memory which should be freed.
    nbytes : int
        Number of bytes of the allocation.

    Examples
    --------
    >>> ptr = huge_alloc(100)
    >>> huge_free(ptr, 100)
    """
    _LIBC.munmap(pointer, nbytes)


_offset: Dict[int, int] = dict()


def alloc_array(shape: Tuple[int, ...],
                dtype: Union[str, np.dtype],
                layout: Tuple[int, ...],
                alignment: int = 0,
                index_to_align: Optional[Tuple[int, ...]] = None,
                alloc: Optional[Callable[[int], int]] = None,
                free: Optional[Callable[[int, int], None]] = None,
                apply_offset: bool = False,
                max_offset: int = 2048) -> np.ndarray:
    """Allocate aligned and padded numpy array.

    Parameters
    ----------
    shape : tuple of int
        Shape of the array.
    dtype : numpy.dtype
        Data type of the array.
    layout : tuple of int
        Layout map. Tuple of ints from `0` to `ndim - 1`, defines the size of
        the strides. The dimension with value `ndim - 1` will have unit stride,
        the dimension with value `0` will have the largest stride.
    alignment : int
        Alignment in bytes.
    index_to_align : tuple of int
        Index of the element that should be aligned (default: first element in
        the array).
    alloc : func
        Memory allocation function accepting a single argument, the size of the
        allocation in bytes (default: `cmalloc`).
    free : func
        Memory freeing function accepting two arguments, a pointer to an
        allocation and the size of the allocation (default: `cfree`).
    apply_offset : bool
        Apply an offset (multiple of alignment) to reduce chance of cache
        conflicts.
    max_offset : int
        Maximum offset to apply. Default is 2KB, half of the normal page size.

    Returns
    -------
    numpy.ndarray
        A numpy.ndarray instance, with memory allocated using `alloc`. Element
        with index `index_to_align` is aligned to `alignment` bytes. Padding is
        applied, such that all but the smallest strides are divisible by
        `alignment` bytes. Memory is unitialized

    Examples
    --------
    Allocating an int32 array with column-major layout:
    >>> x = alloc_array((2, 3), dtype='int32', layout=(1, 0))
    >>> x[:, :] = 0
    >>> x
    array([[0, 0, 0],
           [0, 0, 0]], dtype=int32)
    >>> x.strides
    (4, 8)

    Using layout specification to use row-major layout:
    >>> x = alloc_array((2, 3), dtype='int32', layout=(0, 1))
    >>> x.strides
    (12, 4)

    Use alignment (and thus padding) for an array:
    >>> x = alloc_array((2, 3), dtype='int32', layout=(0, 1), alignment=64)
    >>> x.strides
    (64, 4)

    First element of array should be aligned:
    >>> x.ctypes.data % 64
    0
    """
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

    if alignment not in _offset:
        _offset[alignment] = alignment
    buffer = alloc_buffer(strides_product + alignment + _offset[alignment],
                          alloc, free)
    if alignment:
        pointer_to_align = ctypes.addressof(buffer) + np.sum(
            np.array(strides) * np.array(index_to_align))
        aligned_pointer = (pointer_to_align + alignment -
                           1) // alignment * alignment
        offset = aligned_pointer - pointer_to_align
    else:
        offset = 0
    if apply_offset:
        offset += _offset[alignment]
        _offset[alignment] *= 2
        if _offset[alignment] > max_offset:
            _offset[alignment] = alignment
    return np.ndarray(shape=shape,
                      dtype=dtype,
                      buffer=buffer,
                      offset=offset,
                      strides=strides)


def nbytes(data: np.ndarray) -> int:
    """Return number of data bytes stored in an array buffer.

    In contrast to data.nbytes, this includes padded data and is thus useable
    to determine the number of bytes that need to be copied when using a linear
    1D copy function (e.g. memcpy).

    Parameters
    ----------
    data : numpy.ndarray
        Numpy array.

    Returns
    -------
    int
        Number of bytes in the array buffer, including possible padding. Equals
        the number of bytes that have to be copied with a 1D copy operation.

    Examples
    --------
    >>> nbytes(np.zeros((2, 3), dtype='int32'))
    24
    >>> nbytes(alloc_array((2, 3), dtype='int32', layout=(0, 1)))
    24
    >>> nbytes(alloc_array((2, 3), dtype='int32', layout=(0, 1), alignment=64))
    76
    """
    last_index = np.sum((np.array(data.shape) - 1) * np.array(data.strides))
    return int(last_index + data.itemsize)
