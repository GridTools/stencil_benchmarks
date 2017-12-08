#include "knl/knl_vadv_variant_2d.h"

namespace platform {

    namespace knl {
        template <class Platform, class ValueType>
        void vadv_variant_2d<Platform, ValueType>::vadv() {
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

        template class vadv_variant_2d<knl, float>;
        template class vadv_variant_2d<knl, double>;

    } // namespace knl

} // namespace platform
