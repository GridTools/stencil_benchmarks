#pragma once

#ifdef KNL_NO_HBWMALLOC
#include <cstdlib>
#define KNL_MEMALIGN posix_memalign
#define KNL_FREE free
#else
#include <hbwmalloc.h>
#define KNL_MEMALIGN hbw_posix_memalign
#define KNL_FREE hbw_free
#endif

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
                static std::size_t offset = 64;
                char *raw_ptr;
                if (KNL_MEMALIGN(reinterpret_cast<void **>(&raw_ptr), alignment, n * sizeof(value_type) + offset))
                    throw ERROR("could not allocate HBW memory");

                value_type *ptr = reinterpret_cast<value_type *>(raw_ptr + offset);

                std::size_t *offset_ptr = reinterpret_cast<std::size_t *>(ptr) - 1;
                *offset_ptr = offset;
                if ((offset *= 2) >= 16384)
                    offset = 64;

                return ptr;
            }

            void deallocate(value_type *ptr, std::size_t) {
                std::size_t *offset_ptr = reinterpret_cast<std::size_t *>(ptr) - 1;
                std::size_t offset = *offset_ptr;

                char *raw_ptr = reinterpret_cast<char *>(ptr) - offset;
                KNL_FREE(raw_ptr);
            }
        };

    } // namespace knl

} // namespace platform

#undef KNL_MEMALIGN
#undef KNL_FREE
