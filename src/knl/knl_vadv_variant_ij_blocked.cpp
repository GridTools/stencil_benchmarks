#include "knl/knl_vadv_variant_ij_blocked.h"

namespace platform {

    namespace knl {

        template <class ValueType>
        void vadv_variant_ij_blocked<ValueType>::vadv() {
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
            constexpr int istride = 1;
            const int jstride = this->jstride();
            const int kstride = this->kstride();
            if (this->istride() != 1)
                throw ERROR("this variant is only compatible with unit i-stride layout");

#pragma omp parallel for collapse(2)
            for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                for (int ib = 0; ib < isize; ib += m_iblocksize) {
                    const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                    const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;
                    int index = ib * istride + jb * jstride;

                    for (int j = jb; j < jmax; ++j) {
#pragma omp simd
                        for (int i = ib; i < imax; ++i) {
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
                            index += istride;
                        }
                        index += jstride - (imax - ib) * istride;
                    }
                    index += kstride - (jmax - jb) * jstride;
                }
            }
        }

        template class vadv_variant_ij_blocked<float>;
        template class vadv_variant_ij_blocked<double>;

    } // namespace knl

} // namespace platform
