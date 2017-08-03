#pragma once

#include <thread>

#include "variant.h"

namespace platform {

namespace knl {

template <class Platform, class ValueType>
class knl_variant : public variant<Platform, ValueType> {
 public:
  using base = variant<Platform, ValueType>;

  knl_variant(const arguments_map& args) : base(args) {}

  void prerun() override {
    base::m_dst = base::dst_data() + base::zero_offset();
    base::m_src = base::src_data() + base::zero_offset();

    flush_cache();
  }

  void postrun() override {}

 private:
  void flush_cache() {
#pragma omp parallel
    { std::this_thread::sleep_for(std::chrono::duration<double>(0.02)); }
  }
};

}  // namespace knl

}  // namespace platform
