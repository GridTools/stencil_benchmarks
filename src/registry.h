#pragma once

#include <functional>

#include "arguments.h"

using id = std::tuple<std::string, std::string, std::string>;

using variant_factory =
    factory<variant_base, id_tuple,
            std::function<variant_base*(const arguments_map&)>>;

template <class Type, class... Args>
void register_variant(variant_factory& f, const std::string& platform,
                      const std::string& variant,
                      const std::string& precision) {
  f.add(id(platform, variant, precision),
        [](const arguments_map& args) { return new Type(args); });
}
