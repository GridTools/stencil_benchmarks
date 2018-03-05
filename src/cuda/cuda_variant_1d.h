#pragma once

#include "basic_stencil_variant.h"

namespace platform {

    namespace cuda {

#define LOAD(x) x

#define KERNEL(name, stmt)                                        \
    template <class ValueType>                                    \
    __global__ void kernel_1d_##name(ValueType *__restrict__ dst, \
        const ValueType *__restrict__ src,                        \
        int last,                                                 \
        int istride,                                              \
        int jstride,                                              \
        int kstride) {                                            \
        const int i = blockIdx.x * blockDim.x + threadIdx.x;      \
                                                                  \
        if (i <= last) {                                          \
            stmt;                                                 \
        }                                                         \
    }

        KERNEL(copy, dst[i] = LOAD(src[i]))
        KERNEL(copyi, dst[i] = LOAD(src[i + istride]))
        KERNEL(copyj, dst[i] = LOAD(src[i + jstride]))
        KERNEL(copyk, dst[i] = LOAD(src[i + kstride]))
        KERNEL(avgi, dst[i] = LOAD(src[i - istride]) + LOAD(src[i + istride]))
        KERNEL(avgj, dst[i] = LOAD(src[i - jstride]) + LOAD(src[i + jstride]))
        KERNEL(avgk, dst[i] = LOAD(src[i - kstride]) + LOAD(src[i + kstride]))
        KERNEL(sumi, dst[i] = LOAD(src[i]) + LOAD(src[i + istride]))
        KERNEL(sumj, dst[i] = LOAD(src[i]) + LOAD(src[i + jstride]))
        KERNEL(sumk, dst[i] = LOAD(src[i]) + LOAD(src[i + kstride]))
        KERNEL(lapij,
            dst[i] = LOAD(src[i]) + LOAD(src[i - istride]) + LOAD(src[i + istride]) + LOAD(src[i - jstride]) +
                     LOAD(src[i + jstride]))

#define KERNEL_CALL(name)                                                                         \
    void name(unsigned int i) override {                                                          \
        const int last = this->index(this->isize() - 1, this->jsize() - 1, this->ksize() - 1);    \
        const int blocks = (last + m_blocksize) / m_blocksize;                                    \
        kernel_1d_##name<<<blocks, m_blocksize>>>(                                                \
            this->dst(i), this->src(i), last, this->istride(), this->jstride(), this->kstride()); \
        if (cudaDeviceSynchronize() != cudaSuccess)                                               \
            throw ERROR("error in cudaDeviceSynchronize");                                        \
    }

        template <class ValueType>
        class variant_1d final : public basic_stencil_variant<cuda, ValueType> {
          public:
            using platform = cuda;
            using value_type = ValueType;

            variant_1d(const arguments_map &args)
                : basic_stencil_variant<cuda, ValueType>(args), m_blocksize(args.get<int>("blocksize")) {
                if (m_blocksize <= 0)
                    throw ERROR("invalid block size");
            }

            ~variant_1d() {}

            void prerun() override {
                basic_stencil_variant<platform, value_type>::prerun();

                auto prefetch = [&](const value_type *ptr) {
                    if (cudaMemPrefetchAsync(ptr - this->zero_offset(), this->storage_size() * sizeof(value_type), 0) !=
                        cudaSuccess)
                        throw ERROR("error in cudaMemPrefetchAsync");
                };
                prefetch(this->src());
                prefetch(this->dst());

                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw ERROR("error in cudaDeviceSynchronize");
            }

            KERNEL_CALL(copy)
            KERNEL_CALL(copyi)
            KERNEL_CALL(copyj)
            KERNEL_CALL(copyk)
            KERNEL_CALL(avgi)
            KERNEL_CALL(avgj)
            KERNEL_CALL(avgk)
            KERNEL_CALL(sumi)
            KERNEL_CALL(sumj)
            KERNEL_CALL(sumk)
            KERNEL_CALL(lapij)

          private:
            int m_blocksize;
        };

    } // namespace cuda

} // namespace platform

#undef LOAD
#undef KERNEL
#undef KERNEL_CALL
