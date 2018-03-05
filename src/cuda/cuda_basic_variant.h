#pragma once
#include "basic_stencil_variant.h"

namespace platform {
    namespace cuda {

        template <class ValueType>
        class cuda_basic_variant : public basic_stencil_variant<cuda, ValueType> {
          public:
            using platform = cuda;
            using value_type = ValueType;

            cuda_basic_blocked(const arguments_map &args) : basic_stencil_variant<cuda, ValueType>(args) {}

            ~cuda_basic_blocked() {}

            void setup() override {
                basic_stencil_variant<platform, value_type>::setup();

                auto prefetch = [&](const value_type *ptr) {
                    if (cudaMemPrefetchAsync(ptr - this->zero_offset(), this->storage_size() * sizeof(value_type), 0) !=
                        cudaSuccess)
                        throw ERROR("error in cudaMemPrefetchAsync");
                };
                for (int i = 0; i < this->num_storages_per_field; ++i) {
                    prefetch(this->src(i));
                    prefetch(this->dst(i));
                }

                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw ERROR("error in cudaDeviceSynchronize");
            }
            void teardown() override {
                basic_stencil_variant<platform, value_type>::teardown();

                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw ERROR("error in cudaDeviceSynchronize");
            }
        };
    }
}
