#include "knl/knl_vadv_variant_ij_blocked_k.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        void vadv_variant_ij_blocked_k<Platform, ValueType>::vadv() {
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

#pragma omp parallel
            {
#pragma omp for collapse(2) nowait
                for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                    for (int ib = 0; ib < isize; ib += m_iblocksize) {
                        const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                        const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;

                        for (int j = jb; j < jmax; ++j) {
                            for (int k = 0; k < ksize; ++k) {
#pragma omp simd simdlen(16)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep_k(i,
                                        j,
                                        k,
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
                                }
                            }
                        }
                        for (int j = jb; j < jmax; ++j) {
                            for (int k = ksize - 1; k >= 0; --k) {
#pragma omp simd simdlen(16)
#pragma vector nontemporal(utensstage)
                                for (int i = ib; i < imax; ++i) {
                                    backward_sweep_k(i,
                                        j,
                                        k,
                                        ccol,
                                        dcol,
                                        datacol,
                                        upos,
                                        utensstage,
                                        isize,
                                        jsize,
                                        ksize,
                                        istride,
                                        jstride,
                                        kstride);
                                }
                            }
                        }

                        for (int j = jb; j < jmax; ++j) {
                            for (int k = 0; k < ksize; ++k) {
#pragma omp simd simdlen(16)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep_k(i,
                                        j,
                                        k,
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
                                }
                            }
                        }
                        for (int j = jb; j < jmax; ++j) {
                            for (int k = ksize - 1; k >= 0; --k) {
#pragma omp simd simdlen(16)
#pragma vector nontemporal(vtensstage)
                                for (int i = ib; i < imax; ++i) {
                                    backward_sweep_k(i,
                                        j,
                                        k,
                                        ccol,
                                        dcol,
                                        datacol,
                                        vpos,
                                        vtensstage,
                                        isize,
                                        jsize,
                                        ksize,
                                        istride,
                                        jstride,
                                        kstride);
                                }
                            }
                        }

                        for (int j = jb; j < jmax; ++j) {
                            for (int k = 0; k < ksize; ++k) {
#pragma omp simd simdlen(16)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep_k(i,
                                        j,
                                        k,
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
                                }
                            }
                        }
                        for (int j = jb; j < jmax; ++j) {
                            for (int k = ksize - 1; k >= 0; --k) {
#pragma omp simd simdlen(16)
#pragma vector nontemporal(wtensstage)
                                for (int i = ib; i < imax; ++i) {
                                    backward_sweep_k(i,
                                        j,
                                        k,
                                        ccol,
                                        dcol,
                                        datacol,
                                        wpos,
                                        wtensstage,
                                        isize,
                                        jsize,
                                        ksize,
                                        istride,
                                        jstride,
                                        kstride);
                                }
                            }
                        }
                    }
                }
            }
        }

        template class vadv_variant_ij_blocked_k<flat, float>;
        template class vadv_variant_ij_blocked_k<flat, double>;
        template class vadv_variant_ij_blocked_k<cache, float>;
        template class vadv_variant_ij_blocked_k<cache, double>;

    } // namespace knl

} // namespace platform
