#pragma once

#include "arguments.h"
#include "knl/knl_allocator.h"
#include "variant_base.h"

namespace platform {

    namespace knl {

        struct knl {
            static constexpr const char *name = "knl-flat";

            template <class ValueType>
            using allocator = flat_allocator<ValueType>;

            static void setup(arguments &args);

            static variant_base *create_variant(const arguments_map &args);

            static void flush_cache();
            static void check_cache_conflicts(const std::string &stride_name, std::ptrdiff_t byte_stride);
        };

    } // namespace knl

} // namespace platform
