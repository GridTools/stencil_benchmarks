// Stencil Benchmarks
//
// Copyright (c) 2017-2021, ETH Zurich and MeteoSwiss
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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {
inline std::size_t get_sysinfo(const char *info, std::size_t default_value) {
  int fd = open(info, O_RDONLY);
  if (fd != -1) {
    char buffer[16];
    auto size = read(fd, buffer, sizeof(buffer));
    if (size > 0)
      default_value = std::atoll(buffer);
    close(fd);
  }
  return default_value;
}

inline std::size_t get_meminfo(const char *pattern, std::size_t default_value) {
  auto *fp = std::fopen("/proc/meminfo", "r");
  if (fp) {
    char *line = nullptr;
    size_t line_length;
    while (getline(&line, &line_length, fp) != -1) {
      if (std::sscanf(line, pattern, &default_value) == 1)
        break;
    }
    free(line);
    std::fclose(fp);
  }
  return default_value;
}

inline std::size_t l1_dcache_linesize() {
  static const std::size_t value = get_sysinfo(
      "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size", 64);
  return value;
}

inline std::size_t l1_dcache_sets() {
  static const std::size_t value = get_sysinfo(
      "/sys/devices/system/cpu/cpu0/cache/index0/number_of_sets", 64);
  return value;
}

inline std::size_t page_size() {
  static const std::size_t value = sysconf(_SC_PAGESIZE);
  return value;
}

inline std::size_t hugepage_size() {
  static const std::size_t value =
      get_meminfo("Hugepagesize: %lu kB", 2 * 1024) * 1024;
  return value;
}

struct smallpage_buffer {
  smallpage_buffer(std::size_t size) : size(size) {
    std::size_t paged_size =
        ((size + page_size() - 1) / page_size()) * page_size();
    void *ptr;
    if (posix_memalign(&ptr, page_size(), paged_size))
      throw std::bad_alloc();
    madvise(ptr, paged_size, MADV_NOHUGEPAGE);
    data.reset(ptr);
  }

  std::unique_ptr<void, std::integral_constant<decltype(&std::free), std::free>>
      data;
  std::size_t size;
};

struct mmap_deleter {
  void operator()(void *ptr) const {
    if (munmap(ptr, paged_size)) {
      auto err = errno;
      throw std::runtime_error(std::string("munmap failed with error ") +
                               strerror(err));
    }
  }

  std::size_t paged_size;
};

struct hugepage_buffer {
  hugepage_buffer(std::size_t size, bool transparent) : size(size) {
    std::size_t paged_size =
        ((size + hugepage_size() - 1) / hugepage_size()) * hugepage_size();
    auto flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    if (!transparent)
      flags |= MAP_HUGETLB;
    void *ptr = mmap(nullptr, paged_size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr == MAP_FAILED)
      throw std::bad_alloc();
    if (transparent)
      madvise(ptr, paged_size, MADV_HUGEPAGE);
    data = std::unique_ptr<void, mmap_deleter>(ptr, mmap_deleter{paged_size});
  }

  std::unique_ptr<void, mmap_deleter> data;
  std::size_t size;
};

} // namespace

PYBIND11_MODULE(alloc, m) {
  py::class_<smallpage_buffer>(m, "alloc_smallpages", py::buffer_protocol())
      .def(py::init<std::size_t>())
      .def_buffer([](smallpage_buffer &mb) {
        return py::buffer_info(mb.data.get(), 1,
                               py::format_descriptor<char>::format(), 1,
                               {mb.size}, {1});
      });

  py::class_<hugepage_buffer>(m, "alloc_hugepages", py::buffer_protocol())
      .def(py::init<std::size_t, bool>())
      .def_buffer([](hugepage_buffer &mb) {
        return py::buffer_info(mb.data.get(), 1,
                               py::format_descriptor<char>::format(), 1,
                               {mb.size}, {1});
      });

  m.def("l1_dcache_linesize", l1_dcache_linesize);
  m.def("l1_dcache_sets", l1_dcache_sets);
}
