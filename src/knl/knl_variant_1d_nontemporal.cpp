#include "knl/knl_variant_1d_nontemporal.h"

#define KERNEL(name, stmt)                                                                                   \
    template <class Platform, class ValueType>                                                               \
    void variant_1d_nontemporal<Platform, ValueType>::name() {                                               \
        const int last = this->index(this->isize() - 1, this->jsize() - 1, this->ksize() - 1);               \
        const value_type *__restrict__ src = this->src();                                                    \
        const int istride = this->istride();                                                                 \
        const int jstride = this->jstride();                                                                 \
        const int kstride = this->kstride();                                                                 \
        value_type *__restrict__ dst = this->dst();                                                          \
        _Pragma("omp parallel for simd") _Pragma("vector nontemporal") for (int i = 0; i <= last; ++i) stmt; \
    }

namespace platform {

    namespace knl {

        KERNEL(copy, dst[i] = src[i])
        KERNEL(copyi, dst[i] = src[i + istride])
        KERNEL(copyj, dst[i] = src[i + jstride])
        KERNEL(copyk, dst[i] = src[i + kstride])
        KERNEL(avgi, dst[i] = src[i - istride] + src[i + istride])
        KERNEL(avgj, dst[i] = src[i - jstride] + src[i + jstride])
        KERNEL(avgk, dst[i] = src[i - kstride] + src[i + kstride])
        KERNEL(sumi, dst[i] = src[i] + src[i + istride])
        KERNEL(sumj, dst[i] = src[i] + src[i + jstride])
        KERNEL(sumk, dst[i] = src[i] + src[i + kstride])
        KERNEL(lapij, dst[i] = src[i] + src[i - istride] + src[i + istride] + src[i - jstride] + src[i + jstride])

        template class variant_1d_nontemporal<knl, float>;
        template class variant_1d_nontemporal<knl, double>;

    } // namespace knl

} // namespace platform
