import ctypes

_LIBNUMA = ctypes.cdll.LoadLibrary('libnuma.so')

_LIBNUMA.numa_num_possible_nodes.restype = ctypes.c_int
_LIBNUMA.numa_num_possible_nodes.argtypes = []

_LIBNUMA.numa_alloc_onnode.restype = ctypes.c_void_p
_LIBNUMA.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]

_LIBNUMA.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

_LIBNUMA.numa_bitmask_isbitset.restype = ctypes.c_int
_LIBNUMA.numa_bitmask_isbitset.argtypes = [ctypes.c_void_p, ctypes.c_uint]


def all_nodes():
    nnodes = _LIBNUMA.numa_num_possible_nodes()
    nodes = []
    mask = ctypes.c_void_p.in_dll(_LIBNUMA, 'numa_all_nodes_ptr')
    for node in range(nnodes):
        if _LIBNUMA.numa_bitmask_isbitset(mask, node):
            nodes.append(node)
    return nodes


def alloc_onnode(nbytes, node):
    pointer = _LIBNUMA.numa_alloc_onnode(nbytes, node)
    if not pointer:
        raise RuntimeError(f'could not allocate memory on NUMA node {node}')
    return pointer


def free(pointer, nbytes):
    _LIBNUMA.numa_free(pointer, nbytes)
