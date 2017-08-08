#pragma once

#include <hbwmalloc.h>

#include "except.h"

namespace platform {

    namespace knl {

        template <class ValueType>
        class flat_allocator {
          public:
            using value_type = ValueType;
            static constexpr std::size_t alignment = 64;

            template <class OtherValueType>
            struct rebind {
                using other = flat_allocator<OtherValueType>;
            };

            value_type *allocate(std::size_t n) const {
                value_type *ptr;
                if (hbw_posix_memalign(reinterpret_cast<void **>(&ptr), alignment, n * sizeof(value_type)))
                    throw ERROR("could not allocate HBW memory");
                return ptr;
            }

            void deallocate(value_type *ptr, std::size_t) { hbw_free(ptr); }
        };

    } // namespace knl

} // namespace platform
