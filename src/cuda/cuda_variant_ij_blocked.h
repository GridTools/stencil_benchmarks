#pragma once

#include "basic_stencil_variant.h"

namespace platform {

    namespace cuda {

#define LOAD(x) x

#define KERNEL(name, stmt)                                     \
    template <class ValueType>                                 \
    __global__ void kernel_##name(ValueType *__restrict__ dst, \
        const ValueType *__restrict__ src,                     \
        int isize,                                             \
        int jsize,                                             \
        int ksize,                                             \
        int istride,                                           \
        int jstride,                                           \
        int kstride) {                                         \
        const int i = blockIdx.x * blockDim.x + threadIdx.x;   \
        const int j = blockIdx.y * blockDim.y + threadIdx.y;   \
                                                               \
        int idx = i * istride + j * jstride;                   \
        for (int k = 0; k < ksize; ++k) {                      \
            if (i < isize && j < jsize) {                      \
                stmt;                                          \
                idx += kstride;                                \
            }                                                  \
        }                                                      \
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

#define KERNEL_CALL(name)                                     \
    void name() override {                                    \
        kernel_##name<<<blocks(), blocksize()>>>(this->dst(), \
            this->src(),                                      \
            this->isize(),                                    \
            this->jsize(),                                    \
            this->ksize(),                                    \
            this->istride(),                                  \
            this->jstride(),                                  \
            this->kstride());                                 \
        if (cudaDeviceSynchronize() != cudaSuccess)           \
            throw ERROR("error in cudaDeviceSynchronize");    \
    }

        template <class Platform, class ValueType>
        class variant_ij_blocked final : public basic_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;

            variant_ij_blocked(const arguments_map &args)
                : basic_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
            }

            ~variant_ij_blocked() {}

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
                    (this->jsize() + m_jblocksize - 1) / m_jblocksize);
            }

            dim3 blocksize() const { return dim3(m_iblocksize, m_jblocksize); }

            int m_iblocksize, m_jblocksize;
        };

    } // namespace cuda

} // namespace platform

#undef LOAD
#undef KERNEL
