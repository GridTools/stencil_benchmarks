#pragma once

namespace platform {
    namespace cuda {

        template <class ValueType, template <class, class> class Base>
        class cuda_umesh_variant : public Base<cuda, ValueType> {
          public:
            using platform = cuda;
            using value_type = ValueType;

            cuda_umesh_variant(const arguments_map &args) : Base<cuda, ValueType>(args) {}

            ~cuda_umesh_variant() {}

            void setup() override {
                Base<platform, value_type>::setup();

                auto prefetch = [&](const value_type *ptr) {
                    if (cudaMemPrefetchAsync(ptr - this->zero_offset(), this->storage_size() * sizeof(value_type), 0) !=
                        cudaSuccess)
                        throw ERROR("error in cudaMemPrefetchAsync");
                };
                for (int i = 0; i < this->num_storages_per_field; ++i) {
                    prefetch(this->src(i));
                    prefetch(this->dst(i));
                }
            }
            void teardown() override {
                Base<platform, value_type>::teardown();

                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw ERROR("error in cudaDeviceSynchronize");
            }
        };
    }
}
