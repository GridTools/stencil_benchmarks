#pragma once

#include "knl/knl_platform.h"
#include "knl/knl_vadv_variant.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class vadv_variant_2d final : public knl_vadv_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using platform = Platform;
            using allocator = typename platform::template allocator<value_type>;

            vadv_variant_2d(const arguments_map &args) : knl_vadv_stencil_variant<Platform, ValueType>(args) {}
            ~vadv_variant_2d() {}

            void vadv() override;

          private:
            void kernel_vadv(const int i,
                const int j,
                const value_type *__restrict__ ustage,
                const value_type *__restrict__ upos,
                const value_type *__restrict__ utens,
                value_type *__restrict__ utensstage,
                const value_type *__restrict__ vstage,
                const value_type *__restrict__ vpos,
                const value_type *__restrict__ vtens,
                value_type *__restrict__ vtensstage,
                const value_type *__restrict__ wstage,
                const value_type *__restrict__ wpos,
                const value_type *__restrict__ wtens,
                value_type *__restrict__ wtensstage,
                value_type *__restrict__ ccol,
                value_type *__restrict__ dcol,
                const value_type *__restrict__ wcon,
                value_type *__restrict__ datacol,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {
                this->forward_sweep(i,
                    j,
                    1,
                    0,
                    ccol,
                    dcol,
                    wcon,
                    ustage,
                    upos,
                    utens,
                    utensstage,
                    isize,
                    jsize,
                    ksize,
                    istride,
                    jstride,
                    kstride);
                this->backward_sweep(
                    i, j, ccol, dcol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride);

                this->forward_sweep(i,
                    j,
                    1,
                    0,
                    ccol,
                    dcol,
                    wcon,
                    vstage,
                    vpos,
                    vtens,
                    vtensstage,
                    isize,
                    jsize,
                    ksize,
                    istride,
                    jstride,
                    kstride);
                this->backward_sweep(
                    i, j, ccol, dcol, vpos, vtensstage, isize, jsize, ksize, istride, jstride, kstride);

                this->forward_sweep(i,
                    j,
                    1,
                    0,
                    ccol,
                    dcol,
                    wcon,
                    wstage,
                    wpos,
                    wtens,
                    wtensstage,
                    isize,
                    jsize,
                    ksize,
                    istride,
                    jstride,
                    kstride);
                this->backward_sweep(
                    i, j, ccol, dcol, wpos, wtensstage, isize, jsize, ksize, istride, jstride, kstride);
            }
        };

        extern template class vadv_variant_2d<flat, float>;
        extern template class vadv_variant_2d<flat, double>;
        extern template class vadv_variant_2d<cache, float>;
        extern template class vadv_variant_2d<cache, double>;

    } // knl

} // namespace platform
