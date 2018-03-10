#pragma once

#include "irrumesh_stencil_variant.h"
#include "unstructured/mesh.hpp"
#include "unstructured/neighbours_table.hpp"

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
        int kstride,                                              \
        sneighbours_table table) {                                \
        const int i = blockIdx.x * blockDim.x + threadIdx.x;      \
        const int j = blockIdx.y * blockDim.y + threadIdx.y;      \
                                                                  \
        const int c = j % 2;                                      \
        int idx = i * istride + j * jstride;                      \
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

#define KERNEL_ILP(name, stmt_c0, stmt_c1)                        \
    template <class ValueType>                                    \
    __global__ void kernel_ij_##name(ValueType *__restrict__ dst, \
        const ValueType *__restrict__ src,                        \
        int isize,                                                \
        int jsize,                                                \
        int ksize,                                                \
        int istride,                                              \
        int jstride,                                              \
        int kstride,                                              \
        sneighbours_table table) {                                \
        const int i = blockIdx.x * blockDim.x + threadIdx.x;      \
        const int j = blockIdx.y * blockDim.y + threadIdx.y;      \
                                                                  \
        int idx = i * istride + j * jstride * 2;                  \
        for (int k = 0; k < ksize; ++k) {                         \
            if (i < isize && j < jsize) {                         \
                stmt_c0;                                          \
                stmt_c1;                                          \
                idx += kstride;                                   \
            }                                                     \
        }                                                         \
    }

        KERNEL_ILP(copyu2_ilp, (dst[idx] = LOAD(src[idx])), (dst[idx + jstride] = LOAD(src[idx + jstride])))
        KERNEL(copyu2, (dst[idx] = LOAD(src[idx])), (dst[idx] = LOAD(src[idx])))
//        KERNEL_ILP(on_cells2_ilp,
//            (dst[idx] = LOAD(src[idx + jstride - 1]) + LOAD(src[idx + jstride]) + LOAD(src[idx - jstride])),
//            (dst[idx + jstride] = LOAD(src[idx]) + LOAD(src[idx + 1]) + LOAD(src[idx + jstride * 2])))
//        KERNEL(on_cells2,
//            dst[idx] = LOAD(src[idx + jstride - 1]) + LOAD(src[idx + jstride]) + LOAD(src[idx - jstride]),
//            dst[idx] = LOAD(src[idx - jstride]) + LOAD(src[idx + 1 - jstride]) + LOAD(src[idx + jstride]))

#define KERNEL_CALL(name, blocksmethod)                                 \
    void name(unsigned int t) {                                         \
        kernel_ij_##name<<<blocksmethod(), blocksize()>>>(this->dst(t), \
            this->src(t),                                               \
            this->isize(),                                              \
            this->jsize(),                                              \
            this->ksize(),                                              \
            this->istride(),                                            \
            this->jstride(),                                            \
            this->kstride(),                                            \
            mesh_.get_elements(location::cell).table(location::cell));  \
    }

        template <class ValueType>
        class irregular_umesh final : public cuda_umesh_variant<ValueType, irrumesh_stencil_variant> {
          public:
            using platform = cuda;
            using value_type = ValueType;

            inline irregular_umesh(const arguments_map &args)
                : cuda_umesh_variant<ValueType, irrumesh_stencil_variant>(args),
                  m_iblocksize(args.get<int>("i-blocksize")), m_jblocksize(args.get<int>("j-blocksize")),
                  mesh_(this->isize(), this->jsize(), this->halo()) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
                platform::limit_blocksize(m_iblocksize, m_jblocksize);
                mesh_.print();
                mesh_.test();
            }

            inline ~irregular_umesh() {}

            KERNEL_CALL(copyu2_ilp, blocks_ilp)
            KERNEL_CALL(copyu2, blocks)
            //            KERNEL_CALL(on_cells2_ilp, blocks_ilp)
            //            KERNEL_CALL(on_cells2, blocks)

          private:
            inline dim3 blocks_ilp() const {
                if ((this->jsize() % 2) != 0)
                    throw std::runtime_error("jsize should be a multiple of 2 since umesh contains 2 colors");
                return dim3((this->isize() + m_iblocksize - 1) / m_iblocksize,
                    (this->jsize() / 2 + m_jblocksize - 1) / m_jblocksize);
            }
            inline dim3 blocks() const {
                return dim3((this->isize() + m_iblocksize - 1) / m_iblocksize,
                    (this->jsize() + m_jblocksize - 1) / m_jblocksize);
            }

            inline dim3 blocksize() const { return dim3(m_iblocksize, m_jblocksize); }

            int m_iblocksize, m_jblocksize;
            mesh mesh_;
        };

    } // namespace cuda

} // namespace platform

#undef LOAD
#undef KERNEL_ILP
#undef KERNEL
#undef KERNEL_CALL
