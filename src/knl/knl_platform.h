#pragma once

#include "arguments.h"
#include "knl/knl_allocator.h"
#include "variant_base.h"

namespace platform {

    namespace knl {

        struct knl {
            static constexpr const char *name = "knl";

            template <class ValueType>
            using allocator = flat_allocator<ValueType>;

            static void setup(arguments &args);

            static std::unique_ptr<variant_base> create_variant(const arguments_map &args);

            static void flush_cache();
            static void check_cache_conflicts(const std::string &stride_name, std::ptrdiff_t byte_stride);
        };

    } // namespace knl

    using device = knl::knl;

} // namespace platform
