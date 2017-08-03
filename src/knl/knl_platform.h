#pragma once

#include "arguments.h"

#include "knl/knl_allocator.h"
#include "knl/knl_variant_1d.h"

namespace platform {

namespace knl {

struct flat {
  static constexpr char* name = "knl-flat";

  template <class ValueType>
  using allocator = flat_allocator<ValueType>;

  static void setup(arguments& args);

  static variant_base* create_variant(const arguments_map& args);
};

struct cache {
  static constexpr char* name = "knl-cache";

  template <class ValueType>
  using allocator = std::allocator<ValueType>;

  static void setup(arguments& args);

  static variant_base* create_variant(const arguments_map& args);
};

}  // namespace knl

}  // namespace platform