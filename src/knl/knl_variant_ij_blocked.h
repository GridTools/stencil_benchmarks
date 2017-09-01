#pragma once

#include "knl/knl_basic_stencil_variant.h"

#define KERNEL(name, stmt)                                                                                  \
    void name() override {                                                                                  \
        const value_type *__restrict__ src = this->src();                                                   \
        value_type *__restrict__ dst = this->dst();                                                         \
        const int isize = this->isize();                                                                    \
        const int jsize = this->jsize();                                                                    \
        const int ksize = this->ksize();                                                                    \
        constexpr int istride = 1;                                                                          \
        const int jstride = this->jstride();                                                                \
        const int kstride = this->kstride();                                                                \
        if (this->istride() != 1)                                                                           \
            throw ERROR("this variant is only compatible with unit i-stride layout");                       \
                                                                                                            \
        _Pragma("omp parallel for collapse(2)") for (int jb = 0; jb < jsize; jb += m_jblocksize) {          \
            for (int ib = 0; ib < isize; ib += m_iblocksize) {                                              \
                const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;                    \
                const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;                    \
                int index = ib * istride + jb * jstride;                                                    \
                                                                                                            \
                for (int k = 0; k < ksize; ++k) {                                                           \
                    for (int j = jb; j < jmax; ++j) {                                                       \
                        _Pragma("omp simd") _Pragma("vector nontemporal") for (int i = ib; i < imax; ++i) { \
                            stmt;                                                                           \
                            index += istride;                                                               \
                        }                                                                                   \
                        index += jstride - (imax - ib) * istride;                                           \
                    }                                                                                       \
                    index += kstride - (jmax - jb) * jstride;                                               \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
    }

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class variant_ij_blocked final : public knl_basic_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;

            variant_ij_blocked(const arguments_map &args)
                : knl_basic_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
            }

            KERNEL(copy, dst[index] = src[index])
            KERNEL(copyi, dst[index] = src[index + istride])
            KERNEL(copyj, dst[index] = src[index + jstride])
            KERNEL(copyk, dst[index] = src[index + kstride])
            KERNEL(avgi, dst[index] = src[index - istride] + src[index + istride])
            KERNEL(avgj, dst[index] = src[index - jstride] + src[index + jstride])
            KERNEL(avgk, dst[index] = src[index - kstride] + src[index + kstride])
            KERNEL(sumi, dst[index] = src[index] + src[index + istride])
            KERNEL(sumj, dst[index] = src[index] + src[index + jstride])
            KERNEL(sumk, dst[index] = src[index] + src[index + kstride])
            KERNEL(lapij,
                dst[index] = src[index] + src[index - istride] + src[index + istride] + src[index - jstride] +
                             src[index + jstride])

          private:
            int m_iblocksize, m_jblocksize;
        };

    } // namespace knl

} // namespace platform

#undef KERNEL
