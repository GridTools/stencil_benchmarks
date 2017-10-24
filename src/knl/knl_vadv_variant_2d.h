#pragma once

#include "knl/knl_variant_vadv.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class variant_vadv_2d final : public knl_vadv_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using platform = Platform;
            using allocator = typename platform::template allocator<value_type>;

            variant_vadv_2d(const arguments_map &args) : knl_vadv_stencil_variant<Platform, ValueType>(args) {}
            ~variant_vadv_2d() {}

            void vadv() override {
                const value_type *__restrict__ ustage = this->ustage();
                const value_type *__restrict__ upos = this->upos();
                const value_type *__restrict__ utens = this->utens();
                value_type *__restrict__ utensstage = this->utensstage();
                const value_type *__restrict__ vstage = this->vstage();
                const value_type *__restrict__ vpos = this->vpos();
                const value_type *__restrict__ vtens = this->vtens();
                value_type *__restrict__ vtensstage = this->vtensstage();
                const value_type *__restrict__ wstage = this->wstage();
                const value_type *__restrict__ wpos = this->wpos();
                const value_type *__restrict__ wtens = this->wtens();
                value_type *__restrict__ wtensstage = this->wtensstage();
                value_type *__restrict__ ccol = this->ccol();
                value_type *__restrict__ dcol = this->dcol();
                const value_type *__restrict__ wcon = this->wcon();
                value_type *__restrict__ datacol = this->datacol();
                const int isize = this->isize();
                const int jsize = this->jsize();
                const int ksize = this->ksize();
                const int istride = this->istride();
                const int jstride = this->jstride();
                const int kstride = this->kstride();

                const int last = this->index(isize - 1, jsize - 1, ksize - 1);
#pragma omp parallel for collapse(2)
                for (int j = 0; j < jsize; ++j) {
                    for (int i = 0; i < isize; ++i) {
                        kernel_vadv(i,
                            j,
                            ustage,
                            upos,
                            utens,
                            utensstage,
                            vstage,
                            vpos,
                            vtens,
                            vtensstage,
                            wstage,
                            wpos,
                            wtens,
                            wtensstage,
                            ccol,
                            dcol,
                            wcon,
                            datacol,
                            isize,
                            jsize,
                            ksize,
                            istride,
                            jstride,
                            kstride);
                    }
                }
            }

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

    } // knl

} // namespace platform
