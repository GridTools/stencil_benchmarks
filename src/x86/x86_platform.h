#pragma once

#include "arguments.h"
#include "variant_base.h"

namespace platform {

    namespace x86 {

        struct x86_platform_base {
            static void flush_cache();
            static void check_cache_conflicts(const std::string &stride_name, std::ptrdiff_t byte_stride);
        };

        struct x86_standard : x86_platform_base {
            static constexpr const char *name = "x86-standard";

            template <class ValueType>
            using allocator = std::allocator<ValueType>;

            static void setup(arguments &args);

            static variant_base *create_variant(const arguments_map &args);
        };

    } // namespace x86

} // namespace platform
