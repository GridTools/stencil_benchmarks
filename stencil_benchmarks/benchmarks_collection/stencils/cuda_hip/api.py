import ctypes
import functools
import gc


class Runtime:
    def __init__(self, name, library):
        self.name = name
        self._lib = ctypes.cdll.LoadLibrary(library)

    def _call(self, funcname, argtypes, args):
        funcname = self.name + funcname
        func = getattr(self._lib, funcname)
        func.argtypes = argtypes
        return func(*args)

    def _check_call(self, funcname, argtypes, args):
        if self._call(funcname, argtypes, args) != 0:
            raise RuntimeError(f'GPU runtime function {funcname} failed')

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        if self._call('Malloc', [ctypes.c_void_p, ctypes.c_size_t],
                      [ctypes.byref(ptr), nbytes]) != 0:
            gc.collect()
            self._check_call('Malloc', [ctypes.c_void_p, ctypes.c_size_t],
                             [ctypes.byref(ptr), nbytes])
        return ptr.value

    def free(self, ptr, nbytes=None):
        self._check_call('Free', [ctypes.c_void_p], [ptr])

    def memcpy(self, dst, src, nbytes, kind='Default'):
        kind = {
            'HostToHost': 0,
            'HostToDevice': 1,
            'DeviceToHost': 2,
            'DeviceToDevice': 3,
            'Default': 4
        }[kind]
        self._check_call(
            'Memcpy',
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
            [dst, src, nbytes, kind])

    def device_synchronize(self):
        self._check_call('DeviceSynchronize', [], [])


@functools.lru_cache(maxsize=2)
def runtime(name):
    if name == 'hip':
        return Runtime('hip', 'libhip_hcc.so')
    if name == 'cuda':
        return Runtime('cuda', 'libcudart.so')
    raise RuntimeError('Invalid GPU runtime name')
