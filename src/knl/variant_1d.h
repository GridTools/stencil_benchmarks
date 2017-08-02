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
#pragma omp parallel for
    for (int i = 0; i <= last; ++i) base::m_dst[i] = base::m_src[i];
  }
};

}  // namespace knl

}  // namespace platform
