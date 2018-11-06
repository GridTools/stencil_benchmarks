#include "knl/knl_vadv_variant_ik_blocked_j.h"
#include <thread>
#include <iostream>

namespace platform {

    namespace knl {

        template <class ValueType>
        void vadv_variant_ik_blocked_j<ValueType>::prerun_init() {
            value_type *__restrict__ ustage = this->ustage();
            value_type *__restrict__ upos = this->upos();
            value_type *__restrict__ utens = this->utens();
            value_type *__restrict__ utensstage = this->utensstage();
            value_type *__restrict__ vstage = this->vstage();
            value_type *__restrict__ vpos = this->vpos();
            value_type *__restrict__ vtens = this->vtens();
            value_type *__restrict__ vtensstage = this->vtensstage();
            value_type *__restrict__ wstage = this->wstage();
            value_type *__restrict__ wpos = this->wpos();
            value_type *__restrict__ wtens = this->wtens();
            value_type *__restrict__ wtensstage = this->wtensstage();
            value_type *__restrict__ wcon = this->wcon();
            value_type *__restrict__ ccol = this->ccol();
            value_type *__restrict__ dcol = this->dcol();
            value_type *__restrict__ datacol = this->datacol();
            const int isize = this->isize();
            const int jsize = this->jsize();
            const int ksize = this->ksize();
            const int h = this->halo();

            const value_type dx = 1.0 / isize;
            const value_type dy = 1.0 / jsize;
            const value_type dz = 1.0 / ksize;

            value_type x, y, z;

            auto val = [&](value_type offset1,
                value_type offset2,
                value_type base1,
                value_type base2,
                value_type ispread,
                value_type jspread) {
                return offset1 +
                    base1 * (offset2 + std::cos(M_PI * (ispread * x + ispread * y)) +
                             base2 * std::sin(2 * M_PI * (ispread * x + jspread * y) * z)) /
                             4.0;
            };

 #pragma omp parallel
 #pragma omp for collapse(2) schedule (static, 1) nowait 
            for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                for (int ib = 0; ib < isize; ib += m_iblocksize) {
                    const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                    const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;
                     for (int j = jb; j < jmax; ++j) {
                         for (int i = ib; i < imax; ++i) {
                            for (int k = 0; k < ksize; ++k) {
                                int idx = i + j * isize * ksize + k * isize;
                                x = i * dx;
                                y = j * dy;
                                z = k * dz;
                                ustage[idx] = val(2.2, 1.5, 0.95, 1.18, 18.4, 20.3);
                                upos[idx] = val(3.4, 0.7, 1.07, 1.51, 1.4, 2.3);
                                utens[idx] = val(7.4, 4.3, 1.17, 0.91, 1.4, 2.3);
                                utensstage[idx] = val(3.2, 2.5, 0.95, 1.18, 18.4, 20.3);
                                vstage[idx] = val(2.3, 1.5, 0.95, 1.14, 18.4, 20.3);
                                vpos[idx] = val(3.3, 0.7, 1.07, 1.71, 1.4, 2.3);
                                vtens[idx] = val(7.3, 4.3, 1.17, 0.71, 1.4, 2.3);
                                vtensstage[idx] = val(3.3, 2.4, 0.95, 1.18, 18.4, 20.3);
                                wstage[idx] = val(2.3, 1.5, 0.95, 1.14, 18.4, 20.3);
                                wpos[idx] = val(3.3, 0.7, 1.07, 1.71, 1.4, 2.3);
                                wtens[idx] = val(7.3, 4.3, 1.17, 0.71, 1.4, 2.3);
                                wtensstage[idx] = val(3.3, 2.4, 0.95, 1.18, 18.4, 20.3);
                                wcon[idx] = val(1.3, 0.3, 0.87, 1.14, 1.4, 2.3);
                                ccol[idx] = -1;
                                dcol[idx] = -1;
                                datacol[idx] = -1;
                            }
                        }
                    }
                }
            }  
        }

        template <class ValueType>
        void vadv_variant_ik_blocked_j<ValueType>::vadv() {
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

            constexpr int priority = 3;
            constexpr int prefdist = 4;
            if (this->istride() != 1)
                throw ERROR("this variant is only compatible with unit i-stride layout");

#pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                value_type *__restrict__ ccol_cache = m_ccol_cache.data() + thread_id * m_iblocksize * m_jblocksize;
                value_type *__restrict__ dcol_cache = m_dcol_cache.data() + thread_id * m_iblocksize * m_jblocksize;
#pragma omp for collapse(2) schedule (static, 1) nowait 
                for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                    for (int ib = 0; ib < isize; ib += m_iblocksize) {
                        const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                        const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;

                        for (int j = jb; j < jmax; ++j) {
// forward 0 -- need to split for automatic vectorization
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                            for (int i = ib; i < imax; ++i) {
                                forward_sweep_kmin(i,
                                    j,
                                    1,
                                    0,
                                    ccol,
                                    ccol_cache,
                                    dcol,
                                    dcol_cache,
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
                                    kstride,
                                    ib,
                                    jb);
                            }
                            for (int k = 1; k < ksize - 1; ++k) {
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep_kbody(i,
                                        j,
                                        k,
                                        1,
                                        0,
                                        ccol,
                                        ccol_cache,
                                        dcol,
                                        dcol_cache,
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
                                        kstride,
                                        ib,
                                        jb);
                                }
                            }
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                            for (int i = ib; i < imax; ++i) {
                                forward_sweep_kmax(i,
                                    j,
                                    1,
                                    0,
                                    ccol,
                                    ccol_cache,
                                    dcol,
                                    dcol_cache,
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
                                    kstride,
                                    ib,
                                    jb);
                            }
                            // backward 0
                            for (int k = ksize - 1; k >= 0; --k) {
#pragma omp simd
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
// forward 1
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                            for (int i = ib; i < imax; ++i) {
                                forward_sweep_kmin(i,
                                    j,
                                    0,
                                    1,
                                    ccol,
                                    ccol_cache,
                                    dcol,
                                    dcol_cache,
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
                                    kstride,
                                    ib,
                                    jb);
                            }
                            for (int k = 1; k < ksize - 1; ++k) {
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep_kbody(i,
                                        j,
                                        k,
                                        0,
                                        1,
                                        ccol,
                                        ccol_cache,
                                        dcol,
                                        dcol_cache,
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
                                        kstride,
                                        ib,
                                        jb);
                                }
                            }
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                            for (int i = ib; i < imax; ++i) {
                                forward_sweep_kmax(i,
                                    j,
                                    0,
                                    1,
                                    ccol,
                                    ccol_cache,
                                    dcol,
                                    dcol_cache,
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
                                    kstride,
                                    ib,
                                    jb);
                            }
                            // backward 1
                            for (int k = ksize - 1; k >= 0; --k) {
#pragma omp simd
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
// forward 2
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                            for (int i = ib; i < imax; ++i) {
                                forward_sweep_kmin(i,
                                    j,
                                    0,
                                    0,
                                    ccol,
                                    ccol_cache,
                                    dcol,
                                    dcol_cache,
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
                                    kstride,
                                    ib,
                                    jb);
                            }
                            for (int k = 1; k < ksize - 1; ++k) {
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep_kbody(i,
                                        j,
                                        k,
                                        0,
                                        0,
                                        ccol,
                                        ccol_cache,
                                        dcol,
                                        dcol_cache,
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
                                        kstride,
                                        ib,
                                        jb);
                                }
                            }
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                            for (int i = ib; i < imax; ++i) {
                                forward_sweep_kmax(i,
                                    j,
                                    0,
                                    0,
                                    ccol,
                                    ccol_cache,
                                    dcol,
                                    dcol_cache,
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
                                    kstride,
                                    ib,
                                    jb);
                            }
                            // backward 2
                            for (int k = ksize - 1; k >= 0; --k) {
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

        template class vadv_variant_ik_blocked_j<float>;
        template class vadv_variant_ik_blocked_j<double>;

    } // namespace knl

} // namespace platform
