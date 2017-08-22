#pragma once

#include "knl/knl_basic_stencil_variant.h"

#define KERNEL(name, stmt)                                                                       \
    void name() override {                                                                       \
        const int last = this->index(this->isize() - 1, this->jsize() - 1, this->ksize() - 1);   \
        const value_type *__restrict__ src = this->src();                                        \
        value_type *__restrict__ dst = this->dst();                                              \
        const int istride = this->istride();                                                     \
        const int jstride = this->jstride();                                                     \
        const int kstride = this->kstride();                                                     \
        _Pragma("omp parallel for simd schedule(runtime)") for (int i = 0; i <= last; ++i) stmt; \
    }

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class variant_1d final : public knl_basic_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;

            variant_1d(const arguments_map &args) : knl_basic_stencil_variant<Platform, ValueType>(args) {}

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
        };

    } // namespace knl

} // namespace platform

#undef KERNEL
