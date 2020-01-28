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