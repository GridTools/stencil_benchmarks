#pragma once

#include <cassert>
#include <cstring>

#include "arguments.h"

#include "knl/allocator.h"

#include "knl/variant_1d.h"

namespace platform {

namespace knl {

struct flat {
  static constexpr char* name = "knl-flat";

  template <class ValueType>
  using allocator = knl_flat_allocator<ValueType>;

  static void setup(arguments& args) {
    arguments& pargs = args.command(name, "variant");
    arguments& vargs_1d = pargs.command("1d");
  }

  static variant_base* create_variant(const arguments_map& args) {
    if (args.get("platform") != name) return nullptr;

    if (args.get("precision") == "single") {
      if (args.get("variant") == "1d") return new variant_1d<flat, float>(args);
    } else if (args.get("precision") == "double") {
      if (args.get("variant") == "1d")
        return new variant_1d<flat, double>(args);
    }

    return nullptr;
  }
};

struct cache {
  static constexpr char* name = "knl-cache";

  template <class ValueType>
  using allocator = std::allocator<ValueType>;

  static void setup(arguments& args) {
    arguments& pargs = args.command(name, "variant");
    arguments& vargs_1d = pargs.command("1d");
  }

  static variant_base* create_variant(const arguments_map& args) {
    if (args.get("platform") != name) return nullptr;

    if (args.get("precision") == "single") {
      if (args.get("variant") == "1d")
        return new variant_1d<cache, float>(args);
    } else if (args.get("precision") == "double") {
      if (args.get("variant") == "1d")
        return new variant_1d<cache, double>(args);
    }

    return nullptr;
  }
};

}  // namespace knl

}  // namespace platform
