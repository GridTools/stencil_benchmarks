#pragma once

#include <cuda_runtime.h>

#include "except.h"

namespace platform {

    namespace cuda {

        template < class ValueType >
        class host_allocator {
          public:
            using value_type = ValueType;

            template < class OtherValueType >
            struct rebind {
                using other = host_allocator< OtherValueType >;
            };

            value_type *allocate(std::size_t n) const {
                value_type *ptr;
                if (cudaMallocHost(reinterpret_cast< void ** >(&ptr), n * sizeof(value_type)) != cudaSuccess)
                    throw ERROR("could not allocate pinned memory");
                return ptr;
            }

            void deallocate(value_type *ptr, std::size_t) {
                if (cudaFreeHost(ptr) != cudaSuccess)
                    throw ERROR("could not free memory");
            }
        };

    } // namespace cuda

} // namespace platform
