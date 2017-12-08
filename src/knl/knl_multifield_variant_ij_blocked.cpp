#include "knl/knl_multifield_variant_ij_blocked.h"

#define SRCPTR(fields) SRCPTR##fields

#define SRCPTR1 const value_type *__restrict__ src0 = this->src(0);
#define SRCPTR2 SRCPTR1 const value_type *__restrict__ src1 = this->src(1);
#define SRCPTR3 SRCPTR2 const value_type *__restrict__ src2 = this->src(2);
#define SRCPTR4 SRCPTR3 const value_type *__restrict__ src3 = this->src(3);
#define SRCPTR5 SRCPTR4 const value_type *__restrict__ src4 = this->src(4);
#define SRCPTR6 SRCPTR5 const value_type *__restrict__ src5 = this->src(5);
#define SRCPTR7 SRCPTR6 const value_type *__restrict__ src6 = this->src(6);
#define SRCPTR8 SRCPTR7 const value_type *__restrict__ src7 = this->src(7);
#define SRCPTR9 SRCPTR8 const value_type *__restrict__ src8 = this->src(8);
#define SRCPTR10 SRCPTR9 const value_type *__restrict__ src9 = this->src(9);

#define SRCX(idx, field) src##field[idx]

#define SRC1(idx) SRCX(idx, 0)
#define SRC2(idx) SRC1(idx) + SRCX(idx, 1)
#define SRC3(idx) SRC2(idx) + SRCX(idx, 2)
#define SRC4(idx) SRC3(idx) + SRCX(idx, 3)
#define SRC5(idx) SRC4(idx) + SRCX(idx, 4)
#define SRC6(idx) SRC5(idx) + SRCX(idx, 5)
#define SRC7(idx) SRC6(idx) + SRCX(idx, 6)
#define SRC8(idx) SRC7(idx) + SRCX(idx, 7)
#define SRC9(idx) SRC8(idx) + SRCX(idx, 8)
#define SRC10(idx) SRC9(idx) + SRCX(idx, 9)

#define KERNELF(fields)                                                                                         \
    SRCPTR(fields)                                                                                              \
    value_type *__restrict__ dst = this->dst();                                                                 \
    const int isize = this->isize();                                                                            \
    const int jsize = this->jsize();                                                                            \
    const int ksize = this->ksize();                                                                            \
    constexpr int istride = 1;                                                                                  \
    const int jstride = this->jstride();                                                                        \
    const int kstride = this->kstride();                                                                        \
    if (this->istride() != 1)                                                                                   \
        throw ERROR("this variant is only compatible with unit i-stride layout");                               \
                                                                                                                \
    const int iblocksize = m_iblocksize;                                                                        \
    const int jblocksize = m_jblocksize;                                                                        \
    _Pragma("omp parallel for collapse(2) schedule(static,1)") for (int jb = 0; jb < jsize; jb += jblocksize) { \
        for (int ib = 0; ib < isize; ib += iblocksize) {                                                        \
            const int imax = ib + iblocksize <= isize ? ib + iblocksize : isize;                                \
            const int jmax = jb + jblocksize <= jsize ? jb + jblocksize : jsize;                                \
            int index = ib * istride + jb * jstride;                                                            \
                                                                                                                \
            for (int k = 0; k < ksize; ++k) {                                                                   \
                for (int j = jb; j < jmax; ++j) {                                                               \
                    _Pragma("omp simd") _Pragma("vector nontemporal") for (int i = ib; i < imax; ++i) {         \
                        dst[index] = STMT(SRC##fields);                                                         \
                        index += istride;                                                                       \
                    }                                                                                           \
                    index += jstride - (imax - ib) * istride;                                                   \
                }                                                                                               \
                index += kstride - (jmax - jb) * jstride;                                                       \
            }                                                                                                   \
        }                                                                                                       \
    }

#define KERNEL(name)                                        \
    template <class ValueType>                              \
    void multifield_variant_ij_blocked<ValueType>::name() { \
        const int fields = this->fields();                  \
        if (fields == 1) {                                  \
            KERNELF(1)                                      \
        } else if (fields == 2) {                           \
            KERNELF(2)                                      \
        } else if (fields == 3) {                           \
            KERNELF(3)                                      \
        } else if (fields == 4) {                           \
            KERNELF(4)                                      \
        } else if (fields == 5) {                           \
            KERNELF(5)                                      \
        } else if (fields == 6) {                           \
            KERNELF(6)                                      \
        } else if (fields == 7) {                           \
            KERNELF(7)                                      \
        } else if (fields == 8) {                           \
            KERNELF(8)                                      \
        } else if (fields == 9) {                           \
            KERNELF(9)                                      \
        } else if (fields == 10) {                          \
            KERNELF(10)                                     \
        }                                                   \
    }

namespace platform {

    namespace knl {

#define STMT(src) src(index)
        KERNEL(copy)
#undef STMT
#define STMT(src) src(index + istride)
        KERNEL(copyi)
#undef STMT
#define STMT(src) src(index + jstride)
        KERNEL(copyj)
#undef STMT
#define STMT(src) src(index + kstride)
        KERNEL(copyk)
#undef STMT
#define STMT(src) src(index - istride) + src(index + istride)
        KERNEL(avgi)
#undef STMT
#define STMT(src) src(index - jstride) + src(index + jstride)
        KERNEL(avgj)
#undef STMT
#define STMT(src) src(index - kstride) + src(index + kstride)
        KERNEL(avgk)
#undef STMT
#define STMT(src) src(index) + src(index + istride)
        KERNEL(sumi)
#undef STMT
#define STMT(src) src(index) + src(index + jstride)
        KERNEL(sumj)
#undef STMT
#define STMT(src) src(index) + src(index + kstride)
        KERNEL(sumk)
#undef STMT
#define STMT(src) src(index) + src(index - istride) + src(index + istride) + src(index - jstride) + src(index + jstride)
        KERNEL(lapij)
#undef STMT

        template class multifield_variant_ij_blocked<float>;
        template class multifield_variant_ij_blocked<double>;

    } // namespace knl

} // namespace platform
