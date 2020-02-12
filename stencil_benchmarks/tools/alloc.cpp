#include <cstdlib>
#include <cstring>
#include <memory>

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

struct mmap_deleter {
  void operator()(void *ptr) const {
    // TODO: get correct huge page size (currently assume 2MB)
    auto paged_size = (size / (2 * 1024 * 1024)) * (2 * 1024 * 1024);
    if (munmap(ptr, paged_size)) {
      auto err = errno;
      throw std::runtime_error(std::string("munmap failed with error ") +
                               strerror(err));
    }
  }

  std::size_t size = 0;
};

struct mmap_buffer {
  mmap_buffer(std::size_t size, bool huge_pages) {
    auto flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    if (huge_pages)
      flags |= MAP_HUGETLB;
    void *ptr = mmap(0, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr == MAP_FAILED) {
      auto err = errno;
      throw std::runtime_error(std::string("mmap failed with error ") +
                               strerror(err));
    }
    data = std::unique_ptr<void, mmap_deleter>(ptr, mmap_deleter{size});
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
