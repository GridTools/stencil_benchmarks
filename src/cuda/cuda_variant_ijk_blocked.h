#pragma once

#include "basic_stencil_variant.h"

namespace platform {

    namespace cuda {

#define LOAD(x) x

#define KERNEL(name, stmt)                                         \
    template <class ValueType>                                     \
    __global__ void kernel_ijk_##name(ValueType *__restrict__ dst, \
        const ValueType *__restrict__ src,                         \
        int isize,                                                 \
        int jsize,                                                 \
        int ksize,                                                 \
        int istride,                                               \
        int jstride,                                               \
        int kstride) {                                             \
        const int i = blockIdx.x * blockDim.x + threadIdx.x;       \
        const int j = blockIdx.y * blockDim.y + threadIdx.y;       \
        const int k = blockIdx.z * blockDim.z + threadIdx.z;       \
                                                                   \
        int idx = i * istride + j * jstride + k * kstride;         \
        if (i < isize && j < jsize && k < ksize) {                 \
            stmt;                                                  \
        }                                                          \
    }

        KERNEL(copy, dst[idx] = LOAD(src[idx]))
        KERNEL(copyi, dst[idx] = LOAD(src[idx + istride]))
        KERNEL(copyj, dst[idx] = LOAD(src[idx + jstride]))
        KERNEL(copyk, dst[idx] = LOAD(src[idx + kstride]))
        KERNEL(avgi, dst[idx] = LOAD(src[idx - istride]) + LOAD(src[idx + istride]))
        KERNEL(avgj, dst[idx] = LOAD(src[idx - jstride]) + LOAD(src[idx + jstride]))
        KERNEL(avgk, dst[idx] = LOAD(src[idx - kstride]) + LOAD(src[idx + kstride]))
        KERNEL(sumi, dst[idx] = LOAD(src[idx]) + LOAD(src[idx + istride]))
        KERNEL(sumj, dst[idx] = LOAD(src[idx]) + LOAD(src[idx + jstride]))
        KERNEL(sumk, dst[idx] = LOAD(src[idx]) + LOAD(src[idx + kstride]))
        KERNEL(lapij,
            dst[idx] = LOAD(src[idx]) + LOAD(src[idx - istride]) + LOAD(src[idx + istride]) + LOAD(src[idx - jstride]) +
                       LOAD(src[idx + jstride]))

#define KERNEL_CALL(name)                                         \
    void name(unsigned int i) override {                                        \
        kernel_ijk_##name<<<blocks(), blocksize()>>>(this->dst(i), \
            this->src(i),                                          \
            this->isize(),                                        \
            this->jsize(),                                        \
            this->ksize(),                                        \
            this->istride(),                                      \
            this->jstride(),                                      \
            this->kstride());                                     \
        if (cudaDeviceSynchronize() != cudaSuccess)               \
            throw ERROR("error in cudaDeviceSynchronize");        \
    }

        template <class ValueType>
        class variant_ijk_blocked final : public basic_stencil_variant<cuda, ValueType> {
          public:
            using platform = cuda;
            using value_type = ValueType;

            variant_ijk_blocked(const arguments_map &args)
                : basic_stencil_variant<cuda, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")), m_kblocksize(args.get<int>("k-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0 || m_kblocksize <= 0)
                    throw ERROR("invalid block size");
                platform::limit_blocksize(m_iblocksize, m_jblocksize, m_kblocksize);
            }

            ~variant_ijk_blocked() {}

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
            dim3 blocks() const {
                return dim3((this->isize() + m_iblocksize - 1) / m_iblocksize,
                    (this->jsize() + m_jblocksize - 1) / m_jblocksize,
                    (this->ksize() + m_kblocksize - 1) / m_kblocksize);
            }

            dim3 blocksize() const { return dim3(m_iblocksize, m_jblocksize, m_kblocksize); }

            int m_iblocksize, m_jblocksize, m_kblocksize;
        };

    } // namespace cuda

} // namespace platform

#undef LOAD
#undef KERNEL
#undef KERNEL_CALL
