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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <regex>
#include <string>

#include <sys/mman.h>
#include <unistd.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

struct malloc_buffer {
  malloc_buffer(std::size_t size) : data(std::malloc(size)), size(size) {
    if (!data)
      throw std::bad_alloc();
  }

  std::unique_ptr<void, std::integral_constant<decltype(&std::free), std::free>>
      data;
  std::size_t size;
};

std::size_t get_huge_page_size() {
  std::ifstream meminfo("/proc/meminfo");
  std::regex hugepagesize_regex("Hugepagesize: *([0-9]+) kB$");

  std::string line;
  while (std::getline(meminfo, line)) {
    std::smatch match;
    if (std::regex_match(line, match, hugepagesize_regex))
      return std::stoll(match[1].str()) * 1024;
  }
  return 0;
}

struct mmap_deleter {
  void operator()(void *ptr) const {
    auto paged_size = (size + page_size - 1) / page_size * page_size;
    if (munmap(ptr, paged_size)) {
      auto err = errno;
      throw std::runtime_error(std::string("munmap failed with error ") +
                               strerror(err));
    }
  }

  std::size_t size = 0;
  std::size_t page_size;
};

struct mmap_buffer {
  mmap_buffer(std::size_t size, bool huge_pages) {
    static const std::size_t huge_page_size = get_huge_page_size();
    static const std::size_t normal_page_size = sysconf(_SC_PAGESIZE);
    auto flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    std::size_t page_size = normal_page_size;
    if (huge_pages) {
      if (!huge_page_size)
        throw std::runtime_error("could not get huge page size");
      flags |= MAP_HUGETLB;
      page_size = huge_page_size;
    }
    void *ptr = mmap(0, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr == MAP_FAILED) {
      auto err = errno;
      throw std::runtime_error(std::string("mmap failed with error ") +
                               strerror(err));
    }
    data =
        std::unique_ptr<void, mmap_deleter>(ptr, mmap_deleter{size, page_size});
  }

  std::unique_ptr<void, mmap_deleter> data;
};

} // namespace

PYBIND11_MODULE(alloc, m) {
  py::class_<malloc_buffer>(m, "malloc", py::buffer_protocol())
      .def(py::init<std::size_t>())
      .def_buffer([](malloc_buffer &mb) {
        return py::buffer_info(mb.data.get(), 1,
                               py::format_descriptor<char>::format(), 1,
                               {mb.size}, {1});
      });

  py::class_<mmap_buffer>(m, "mmap", py::buffer_protocol())
      .def(py::init<std::size_t, bool>())
      .def_buffer([](mmap_buffer &mb) {
        return py::buffer_info(mb.data.get(), 1,
                               py::format_descriptor<char>::format(), 1,
                               {mb.data.get_deleter().size}, {1});
      });

  m.def("l1_dcache_size", []() { return sysconf(_SC_LEVEL1_DCACHE_SIZE); });

  m.def("l1_dcache_linesize",
        []() { return sysconf(_SC_LEVEL1_DCACHE_LINESIZE); });

  m.def("l1_dcache_assoc", []() { return sysconf(_SC_LEVEL1_DCACHE_ASSOC); });
}
