#include "knl/knl_vadv_variant_ij_blocked_k_ring.h"

namespace platform {

    namespace knl {

        template <class ValueType>
            void vadv_variant_ij_blocked_k_ring<ValueType>::vadv() {
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
                                //example of software prefetching using asm or gcc builtin
                                if (j + prefdist < jmax){
                                      const int prefindex = ib * istride + (j + prefdist)* jstride;
                                      asm volatile("prfm PSTL3KEEP, [%[ptr]]\n" ::[ptr] "r"(ccol[prefindex]) :);
                                      asm volatile("prfm PSTL3KEEP, [%[ptr]]\n" ::[ptr] "r"(dcol[prefindex]) :);
                                      asm volatile("prfm PLDL3KEEP, [%[ptr]]\n" ::[ptr] "r"(wcon[prefindex + kstride]) :);
                                      asm volatile("prfm PLDL3KEEP, [%[ptr]]\n" ::[ptr] "r"(ustage[prefindex + kstride]) :);
                                      asm volatile("prfm PLDL3KEEP, [%[ptr]]\n" ::[ptr] "r"(utens[prefindex]) :);
                                      asm volatile("prfm PLDL3KEEP, [%[ptr]]\n" ::[ptr] "r"(utensstage[prefindex]) :);
                                      asm volatile("prfm PLDL3KEEP, [%[ptr]]\n" ::[ptr] "r"(upos[prefindex]) :);
                                }
                                    /*__builtin_prefetch(&ccol[prefindex], 1, priority);
                                      __builtin_prefetch(&dcol[prefindex], 1, priority);
                                      __builtin_prefetch(&wcon[prefindex + kstride], 0, priority);
                                      __builtin_prefetch(&wcon[prefindex + kstride], 0, priority);
                                      __builtin_prefetch(&ustage[prefindex + kstride], 0, priority);
                                      __builtin_prefetch(&utens[prefindex], 0, priority);
                                      __builtin_prefetch(&utensstage[prefindex], 0, priority);
                                      __builtin_prefetch(&upos[prefindex], 0, priority);
                                } */

                                //forward 0 -- need to split for automatic vectorization
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep_kmin(i, j, 1, 0, ccol, ccol_cache, dcol, dcol_cache,
                                            wcon, ustage, upos, utens, utensstage,
                                            isize, jsize, ksize, istride, jstride, kstride,
                                            ib, jb);
                                }
                                for (int k = 1; k < ksize - 1; ++k) {
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                    for (int i = ib; i < imax; ++i) {
                                        forward_sweep_kbody(i, j, k, 1, 0, ccol, ccol_cache, dcol, dcol_cache,
                                                wcon, ustage, upos, utens, utensstage,
                                                isize, jsize, ksize, istride, jstride, kstride,
                                                ib, jb);
                                    }
                                }
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep_kmax(i, j, 1, 0, ccol, ccol_cache, dcol, dcol_cache,
                                            wcon, ustage, upos, utens, utensstage, 
                                            isize, jsize, ksize, istride, jstride, kstride, 
                                            ib, jb); 
                                }
                                //backward 0
                                for (int k = ksize - 1; k >= 0; --k) {
#pragma omp simd
#pragma vector nontemporal(utensstage)
                                    for (int i = ib; i < imax; ++i) {
                                        backward_sweep_k(i, j, k, ccol, dcol, datacol, upos, utensstage,
                                                isize, jsize, ksize, istride, jstride, kstride);
                                    }
                                }
                                //forward 1
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep_kmin(i, j, 0, 1, ccol, ccol_cache, dcol, dcol_cache,
                                            wcon, vstage, vpos, vtens, vtensstage, 
                                            isize, jsize, ksize, istride, jstride, kstride, 
                                            ib, jb);
                                }
                                for (int k = 1; k < ksize - 1; ++k) {
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                    for (int i = ib; i < imax; ++i) {
                                        forward_sweep_kbody(i, j, k, 0, 1, ccol, ccol_cache, dcol, dcol_cache, 
                                                wcon, vstage, vpos, vtens, vtensstage, 
                                                isize, jsize, ksize, istride, jstride, kstride, 
                                                ib, jb); 
                                    }
                                }
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep_kmax(i, j, 0, 1, ccol, ccol_cache, dcol, dcol_cache,
                                            wcon, vstage, vpos, vtens, vtensstage, 
                                            isize, jsize, ksize, istride, jstride, kstride, 
                                            ib, jb); 
                                }
                                //backward 1
                                for (int k = ksize - 1; k >= 0; --k) {
#pragma omp simd
#pragma vector nontemporal(vtensstage)
                                    for (int i = ib; i < imax; ++i) {
                                        backward_sweep_k(i, j, k, ccol, dcol, datacol, vpos, vtensstage,
                                                isize, jsize, ksize, istride, jstride, kstride);
                                    }
                                }
                                //forward 2 
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep_kmin(i, j, 0, 0, ccol, ccol_cache, dcol, dcol_cache,  
                                            wcon, wstage, wpos, wtens, wtensstage, 
                                            isize, jsize, ksize, istride, jstride, kstride, 
                                            ib, jb); 
                                }
                                for (int k = 1; k < ksize - 1; ++k) {
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                    for (int i = ib; i < imax; ++i) {
                                        forward_sweep_kbody(i, j, k, 0, 0, ccol, ccol_cache, dcol, dcol_cache, 
                                                wcon, wstage, wpos, wtens, wtensstage, 
                                                isize, jsize, ksize, istride, jstride, kstride, 
                                                ib, jb); 
                                    }
                                }
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep_kmax(i, j, 0, 0, ccol, ccol_cache, dcol, dcol_cache,
                                            wcon, wstage, wpos, wtens, wtensstage, 
                                            isize, jsize, ksize, istride, jstride, kstride, 
                                            ib, jb); 
                                }
                                //backward 2
                                for (int k = ksize - 1; k >= 0; --k) {
#pragma vector nontemporal(wtensstage)
                                    for (int i = ib; i < imax; ++i) {
                                        backward_sweep_k(i, j, k, ccol, dcol, datacol, wpos, wtensstage,
                                                isize, jsize, ksize, istride, jstride, kstride);
                                    }
                                }
                            }
                        }
                    }
                }
            }

        template class vadv_variant_ij_blocked_k_ring<float>;
        template class vadv_variant_ij_blocked_k_ring<double>;

    } // namespace knl

} // namespace platform
