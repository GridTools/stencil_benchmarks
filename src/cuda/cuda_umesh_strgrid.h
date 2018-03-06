#pragma once

#include "cuda_umesh_variant.h"

namespace platform {

    namespace cuda {

#define LOAD(x) x

#define KERNEL(name, stmt_c0, stmt_c1)                            \
    template <class ValueType>                                    \
    __global__ void kernel_ij_##name(ValueType *__restrict__ dst, \
        const ValueType *__restrict__ src,                        \
        int isize,                                                \
        int jsize,                                                \
        int ksize,                                                \
        int istride,                                              \
        int jstride,                                              \
        int kstride) {                                            \
        const int i = blockIdx.x * blockDim.x + threadIdx.x;      \
        const int j = blockIdx.y * blockDim.y + threadIdx.y;      \
                                                                  \
        const int c = j % 2;                                      \
                                                                  \
        int idx = i * istride + j * jstride * 2;                  \
        if (c == 0)                                               \
            for (int k = 0; k < ksize; ++k) {                     \
                if (i < isize && j < jsize) {                     \
                    stmt_c0;                                      \
                    idx += kstride;                               \
                }                                                 \
            }                                                     \
        else                                                      \
            for (int k = 0; k < ksize; ++k) {                     \
                if (i < isize && j < jsize) {                     \
                    stmt_c1;                                      \
                    idx += kstride;                               \
                }                                                 \
            }                                                     \
    }

#define KERNEL_ILP(name, stmt)                                    \
    template <class ValueType>                                    \
    __global__ void kernel_ij_##name(ValueType *__restrict__ dst, \
        const ValueType *__restrict__ src,                        \
        int isize,                                                \
        int jsize,                                                \
        int ksize,                                                \
        int istride,                                              \
        int jstride,                                              \
        int kstride) {                                            \
        const int i = blockIdx.x * blockDim.x + threadIdx.x;      \
        const int j = blockIdx.y * blockDim.y + threadIdx.y;      \
                                                                  \
        int idx = i * istride + j * jstride * 2;                  \
        for (int k = 0; k < ksize; ++k) {                         \
            if (i < isize && j < jsize) {                         \
                stmt;                                             \
                idx += kstride;                                   \
            }                                                     \
        }                                                         \
    }

        KERNEL(copyu_ilp, (dst[idx] = LOAD(src[idx])), (dst[idx + jstride] = LOAD(src[idx + jstride])))
        KERNEL_ILP(copyu, dst[idx] = LOAD(src[idx]))
        KERNEL_ILP(on_cells_ilp, dst[idx] = LOAD(src[idx]); dst[idx + jstride] = LOAD(src[idx + jstride]))
        KERNEL(on_cells,
            (dst[idx] = LOAD(src[idx + jstride - 1]) + LOAD(src[idx + jstride]) + LOAD(src[idx - jstride])),
            (dst[idx + jstride] = LOAD(src[idx]) + LOAD(src[idx + 1]) + LOAD(src[idx + jstride * 2])))

#define KERNEL_CALL(name, blocksmethod)                                 \
    void name(unsigned int t) {                                         \
        kernel_ij_##name<<<blocksmethod(), blocksize()>>>(this->dst(t), \
            this->src(t),                                               \
            this->isize(),                                              \
            this->jsize(),                                              \
            this->ksize(),                                              \
            this->istride(),                                            \
            this->jstride(),                                            \
            this->kstride());                                           \
    }

        template <class ValueType>
        class umesh_strgrid final : public cuda_umesh_variant<ValueType> {
          public:
            using platform = cuda;
            using value_type = ValueType;

            inline umesh_strgrid(const arguments_map &args)
                : cuda_umesh_variant<ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
                platform::limit_blocksize(m_iblocksize, m_jblocksize);
            }

            inline ~umesh_strgrid() {}

            KERNEL_CALL(copyu_ilp, blocks_ilp)
            KERNEL_CALL(copyu, blocks)
            KERNEL_CALL(on_cells_ilp, blocks_ilp)
            KERNEL_CALL(on_cells, blocks)

          private:
            inline dim3 blocks_ilp() const {
                if ((this->jsize() % 2) != 0)
                    throw std::runtime_error("jsize should be a multiple of 2 since umesh contains 2 colors");
                return dim3((this->isize() + m_iblocksize - 1) / m_iblocksize,
                    (this->jsize() / 2 + m_jblocksize - 1) / m_jblocksize);
            }
            inline dim3 blocks() const {
                if ((this->jsize() % 2) != 0)
                    throw std::runtime_error("jsize should be a multiple of 2 since umesh contains 2 colors");
                return dim3((this->isize() + m_iblocksize - 1) / m_iblocksize,
                    (this->jsize() / 2 + m_jblocksize - 1) / m_jblocksize);
            }

            inline dim3 blocksize() const { return dim3(m_iblocksize, m_jblocksize); }

            int m_iblocksize, m_jblocksize;
        };

    } // namespace cuda

} // namespace platform

#undef LOAD
#undef KERNEL_ILP
#undef KERNEL
#undef KERNEL_CALL
