// Stencil Benchmarks
//
// Copyright (c) 2017-2020, ETH Zurich and MeteoSwiss
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <functional>
#include <random>
#include <thread>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {
template <class T> void random_fill(T *first, T *last) {
  py::gil_scoped_release release;
  std::vector<std::thread> threads;
  std::size_t nthreads = std::thread::hardware_concurrency();
  for (std::size_t i = 0; i < nthreads; ++i) {
    threads.emplace_back([=]() {
      std::random_device seed;
      auto rand = std::bind(std::uniform_real_distribution<T>(),
                            std::minstd_rand(seed()));

      std::size_t n = std::distance(first, last);
      T *thread_first = first + i * n / nthreads;
      T *thread_last = first + (i + 1) * n / nthreads;
      std::generate(thread_first, thread_last, rand);
    });
  }
  for (auto &thread : threads)
    thread.join();
}
} // namespace

PYBIND11_MODULE(parallel, m) {
  m.def("random_fill", [](py::buffer b) {
    py::buffer_info info = b.request();

    char *first = reinterpret_cast<char *>(info.ptr);
    char *last = first + info.itemsize;
    for (int dim = 0; dim < info.ndim; ++dim)
      last += (info.shape[dim] - 1) * info.strides[dim];

    if (info.format == py::format_descriptor<float>::format()) {
      random_fill<float>(reinterpret_cast<float *>(first),
                         reinterpret_cast<float *>(last));
    } else if (info.format == py::format_descriptor<double>::format()) {
      random_fill<double>(reinterpret_cast<double *>(first),
                          reinterpret_cast<double *>(last));
    } else {
      throw std::runtime_error("data type not supported");
    }
  });
}