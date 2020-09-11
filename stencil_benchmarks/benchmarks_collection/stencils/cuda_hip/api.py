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
import ctypes
import functools
import gc
import weakref


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

        ptr = ptr.value
        buffer = (ctypes.c_byte * nbytes).from_address(ptr)

        def free(p):
            self._check_call('Free', [ctypes.c_void_p], [p])

        weakref.finalize(buffer, free, ptr)
        return buffer

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
        return Runtime('hip', 'libamdhip64.so')
    if name == 'cuda':
        return Runtime('cuda', 'libcudart.so')
    raise RuntimeError('Invalid GPU runtime name')
