#pragma once

#include <cuda_runtime.h>

#include "except.h"
#include "variant.h"

namespace platform {

    namespace cuda {

        template < class Platform, class ValueType >
        class cuda_variant : public variant< Platform, ValueType > {
          public:
            using value_type = ValueType;

            cuda_variant(const arguments_map &args) : variant< Platform, ValueType >(args) {
                value_type *d_src, *d_dst;
                if (cudaMalloc(reinterpret_cast< void ** >(&d_src), this->storage_size() * sizeof(value_type)) !=
                    cudaSuccess)
                    throw ERROR("could not allocate device memory");
                if (cudaMalloc(reinterpret_cast< void ** >(&d_dst), this->storage_size() * sizeof(value_type)) !=
                    cudaSuccess)
                    throw ERROR("could not allocate device memory");
                this->set_ptrs(d_src + this->zero_offset(), d_dst + this->zero_offset());
            }

            virtual ~cuda_variant() {}

            void prerun() override {
                void *src_ptr = this->src() - this->zero_offset();
                void *dst_ptr = this->dst() - this->zero_offset();

                if (cudaMemcpy(
                        src_ptr, this->src_data(), this->storage_size() * sizeof(value_type), cudaMemcpyHostToDevice) !=
                    cudaSuccess)
                    throw ERROR("could not copy data to device");
                if (cudaMemcpy(
                        dst_ptr, this->dst_data(), this->storage_size() * sizeof(value_type), cudaMemcpyHostToDevice) !=
                    cudaSuccess)
                    throw ERROR("could not copy data to device");
            }

            void postrun() override {
                void *src_ptr = this->src() - this->zero_offset();
                void *dst_ptr = this->dst() - this->zero_offset();

                if (cudaMemcpy(
                        this->src_data(), src_ptr, this->storage_size() * sizeof(value_type), cudaMemcpyDeviceToHost) !=
                    cudaSuccess)
                    throw ERROR("could not copy data from device");
                if (cudaMemcpy(
                        this->dst_data(), dst_ptr, this->storage_size() * sizeof(value_type), cudaMemcpyDeviceToHost) !=
                    cudaSuccess)
                    throw ERROR("could not copy data from device");
            }
        };

    } // namespace cuda

} // namespace platform
