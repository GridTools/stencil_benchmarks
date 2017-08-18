#pragma once

#include "arguments.h"
#include "knl/knl_allocator.h"
#include "variant_base.h"

namespace platform {

    namespace knl {

        struct knl_platform_base {
            static void flush_cache();
            static void check_cache_conflicts(const std::string &stride_name, std::ptrdiff_t byte_stride);
        };

        struct flat : knl_platform_base {
            static constexpr const char *name = "knl-flat";

            template <class ValueType>
            using allocator = flat_allocator<ValueType>;

            static void setup(arguments &args);

            static variant_base *create_variant(const arguments_map &args);
        };

        struct cache : knl_platform_base {
            static constexpr const char *name = "knl-cache";

            template <class ValueType>
            using allocator = std::allocator<ValueType>;

            static void setup(arguments &args);

            static variant_base *create_variant(const arguments_map &args);
        };

    } // namespace knl

} // namespace platform
