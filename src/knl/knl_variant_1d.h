#pragma once

#include "knl/knl_variant.h"

namespace platform {

namespace knl {

template <class Platform, class ValueType>
class variant_1d final : public knl_variant<Platform, ValueType> {
 public:
  using base = knl_variant<Platform, ValueType>;

  variant_1d(const arguments_map& args)
      : knl_variant<Platform, ValueType>(args) {}

  void copy() override {
    const int last =
        base::index(base::isize() - 1, base::jsize() - 1, base::ksize() - 1);
#pragma omp parallel for simd
    for (int i = 0; i <= last; ++i) base::m_dst[i] = base::m_src[i];
  }

  void copyi() override {
    const int last =
        base::index(base::isize() - 1, base::jsize() - 1, base::ksize() - 1);
    const int istride = base::istride();
#pragma omp parallel for simd
    for (int i = 0; i <= last; ++i) base::m_dst[i] = base::m_src[i + istride];
  }

  void copyj() override {
    const int last =
        base::index(base::isize() - 1, base::jsize() - 1, base::ksize() - 1);
    const int jstride = base::jstride();
#pragma omp parallel for simd
    for (int i = 0; i <= last; ++i) base::m_dst[i] = base::m_src[i + jstride];
  }

  void copyk() override {
    const int last =
        base::index(base::isize() - 1, base::jsize() - 1, base::ksize() - 1);
    const int kstride = base::kstride();
#pragma omp parallel for simd
    for (int i = 0; i <= last; ++i) base::m_dst[i] = base::m_src[i + kstride];
  }

  void avgi() override {
    const int last =
        base::index(base::isize() - 1, base::jsize() - 1, base::ksize() - 1);
    const int istride = base::istride();
#pragma omp parallel for simd
    for (int i = 0; i <= last; ++i)
      base::m_dst[i] = base::m_src[i - istride] + base::m_src[i + istride];
  }

  void avgj() override {
    const int last =
        base::index(base::isize() - 1, base::jsize() - 1, base::ksize() - 1);
    const int jstride = base::jstride();
#pragma omp parallel for simd
    for (int i = 0; i <= last; ++i)
      base::m_dst[i] = base::m_src[i - jstride] + base::m_src[i + jstride];
  }

  void avgk() override {
    const int last =
        base::index(base::isize() - 1, base::jsize() - 1, base::ksize() - 1);
    const int kstride = base::kstride();
#pragma omp parallel for simd
    for (int i = 0; i <= last; ++i)
      base::m_dst[i] = base::m_src[i - kstride] + base::m_src[i + kstride];
  }

  void sumi() override {
    const int last =
        base::index(base::isize() - 1, base::jsize() - 1, base::ksize() - 1);
    const int istride = base::istride();
#pragma omp parallel for simd
    for (int i = 0; i <= last; ++i)
      base::m_dst[i] = base::m_src[i] + base::m_src[i + istride];
  }

  void sumj() override {
    const int last =
        base::index(base::isize() - 1, base::jsize() - 1, base::ksize() - 1);
    const int jstride = base::jstride();
#pragma omp parallel for simd
    for (int i = 0; i <= last; ++i)
      base::m_dst[i] = base::m_src[i] + base::m_src[i + jstride];
  }

  void sumk() override {
    const int last =
        base::index(base::isize() - 1, base::jsize() - 1, base::ksize() - 1);
    const int kstride = base::kstride();
#pragma omp parallel for simd
    for (int i = 0; i <= last; ++i)
      base::m_dst[i] = base::m_src[i] + base::m_src[i + kstride];
  }

  void lapij() override {
    const int last =
        base::index(base::isize() - 1, base::jsize() - 1, base::ksize() - 1);
    const int istride = base::istride();
    const int jstride = base::jstride();
#pragma omp parallel for simd
    for (int i = 0; i <= last; ++i) {
      base::m_dst[i] = base::m_src[i] + base::m_src[i - istride] +
                       base::m_src[i + istride] + base::m_src[i - jstride] +
                       base::m_src[i + jstride];
    }
  }
};

}  // namespace knl

}  // namespace platform
