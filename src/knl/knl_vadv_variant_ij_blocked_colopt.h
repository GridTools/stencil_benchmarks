#pragma once

#include <omp.h>

#include "knl/knl_variant_vadv.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class variant_vadv_ij_blocked_colopt final : public knl_vadv_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using platform = Platform;

            variant_vadv_ij_blocked_colopt(const arguments_map &args)
                : knl_vadv_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
                std::cout << m_iblocksize << " " << m_jblocksize << std::endl;
            }
            ~variant_vadv_ij_blocked_colopt() {}

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
                constexpr int istride = 1;
                const int jstride = this->jstride();
                const int kstride = this->kstride();
                if (this->istride() != 1)
                    throw ERROR("this variant is only compatible with unit i-stride layout");

#pragma omp parallel
                {
                    const int thread_index = omp_get_thread_num();
#pragma omp for collapse(2) nowait
                    for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                        for (int ib = 0; ib < isize; ib += m_iblocksize) {
                            const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                            const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;
                            int index = ib * istride + jb * jstride;

                            for (int j = jb; j < jmax; ++j) {
#pragma omp simd safelen(8)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep(i,
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
                                        kstride,
                                        thread_index);
                                    backward_sweep(i,
                                        j,
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
                                        kstride,
                                        thread_index);
                                    index += istride;
                                }
                                index += jstride - (imax - ib) * istride;
                            }
                            index += kstride - (jmax - jb) * jstride;
                        }
                    }
#pragma omp for collapse(2) nowait
                    for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                        for (int ib = 0; ib < isize; ib += m_iblocksize) {
                            const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                            const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;
                            int index = ib * istride + jb * jstride;

                            for (int j = jb; j < jmax; ++j) {
#pragma omp simd safelen(8)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep(i,
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
                                        kstride,
                                        thread_index);
                                    backward_sweep(i,
                                        j,
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
                                        kstride,
                                        thread_index);
                                    index += istride;
                                }
                                index += jstride - (imax - ib) * istride;
                            }
                            index += kstride - (jmax - jb) * jstride;
                        }
                    }
#pragma omp for collapse(2)
                    for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                        for (int ib = 0; ib < isize; ib += m_iblocksize) {
                            const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                            const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;
                            int index = ib * istride + jb * jstride;

                            for (int j = jb; j < jmax; ++j) {
#pragma omp simd safelen(8)
                                for (int i = ib; i < imax; ++i) {
                                    forward_sweep(i,
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
                                        kstride,
                                        thread_index);
                                    backward_sweep(i,
                                        j,
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
                                        kstride,
                                        thread_index);
                                    index += istride;
                                }
                                index += jstride - (imax - ib) * istride;
                            }
                            index += kstride - (jmax - jb) * jstride;
                        }
                    }
                }
            }

          private:
#pragma omp declare simd linear(i) uniform( \
    j, ccol, dcol, datacol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            /*__attribute__((always_inline))*/ void backward_sweep(const int i,
                const int j,
                const value_type *__restrict__ ccol,
                const value_type *__restrict__ dcol,
                value_type *__restrict__ datacol,
                const value_type *__restrict__ upos,
                value_type *__restrict__ utensstage,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride,
                const int thread_index) {
                constexpr value_type dtr_stage = 3.0 / 20.0;
                const int kcolstride = kstride;

                // k maximum
                {
                    const int k = ksize - 1;
                    const int index = i * istride + j * jstride + k * kstride;
                    const int colindex = i * istride + thread_index * jstride + k * kstride;
                    datacol[colindex] = dcol[colindex];
                    utensstage[index] = dtr_stage * (datacol[colindex] - upos[index]);
                }

                // k body
                for (int k = ksize - 2; k >= 0; --k) {
                    const int index = i * istride + j * jstride + k * kstride;
                    const int colindex = i * istride + thread_index * jstride + k * kstride;
                    datacol[colindex] = dcol[colindex] - ccol[colindex] * datacol[colindex + kcolstride];
                    utensstage[index] = dtr_stage * (datacol[colindex] - upos[index]);
                }
            }

#pragma omp declare simd linear(i) uniform( \
    j, ishift, jshift, ccol, wcon, ustage, upos, utens, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            /*__attribute__((always_inline))*/ void forward_sweep(const int i,
                const int j,
                const int ishift,
                const int jshift,
                value_type *__restrict__ ccol,
                value_type *__restrict__ dcol,
                const value_type *__restrict__ wcon,
                const value_type *__restrict__ ustage,
                const value_type *__restrict__ upos,
                const value_type *__restrict__ utens,
                const value_type *__restrict__ utensstage,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride,
                const int thread_index) {
                constexpr value_type dtr_stage = 3.0 / 20.0;
                constexpr value_type beta_v = 0;
                constexpr value_type bet_m = 0.5 * (1.0 - beta_v);
                constexpr value_type bet_p = 0.5 * (1.0 + beta_v);
                const int kcolstride = kstride;

                // k minimum
                {
                    const int k = 0;
                    const int index = i * istride + j * jstride + k * kstride;
                    const int colindex = i * istride + thread_index * jstride + k * kstride;
                    value_type gcv = value_type(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride] +
                                                            wcon[index + kstride]);
                    value_type cs = gcv * bet_m;

                    ccol[colindex] = gcv * bet_p;
                    value_type bcol = dtr_stage - ccol[colindex];

                    value_type correction_term = -cs * (ustage[index + kstride] - ustage[index]);
                    dcol[colindex] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    value_type divided = value_type(1.0) / bcol;
                    ccol[colindex] = ccol[colindex] * divided;
                    dcol[colindex] = dcol[colindex] * divided;
                }

                // k body
                for (int k = 1; k < ksize - 1; ++k) {
                    const int index = i * istride + j * jstride + k * kstride;
                    const int colindex = i * istride + thread_index * jstride + k * kstride;
                    value_type gav =
                        value_type(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);
                    value_type gcv = value_type(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride] +
                                                            wcon[index + kstride]);

                    value_type as = gav * bet_m;
                    value_type cs = gcv * bet_m;

                    value_type acol = gav * bet_p;
                    ccol[colindex] = gcv * bet_p;
                    value_type bcol = dtr_stage - acol - ccol[colindex];

                    value_type correction_term = -as * (ustage[index - kstride] - ustage[index]) -
                                                 cs * (ustage[index + kstride] - ustage[index]);
                    dcol[colindex] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    value_type divided = value_type(1.0) / (bcol - ccol[colindex - kcolstride] * acol);
                    ccol[colindex] = ccol[colindex] * divided;
                    dcol[colindex] = (dcol[colindex] - dcol[colindex - kcolstride] * acol) * divided;
                }

                // k maximum
                {
                    const int k = ksize - 1;
                    const int index = i * istride + j * jstride + k * kstride;
                    const int colindex = i * istride + thread_index * jstride + k * kstride;
                    value_type gav =
                        value_type(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);

                    value_type as = gav * bet_m;

                    value_type acol = gav * bet_p;
                    value_type bcol = dtr_stage - acol;

                    value_type correction_term = -as * (ustage[index - kstride] - ustage[index]);
                    dcol[colindex] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    value_type divided = value_type(1.0) / (bcol - ccol[colindex - kcolstride] * acol);
                    dcol[colindex] = (dcol[colindex] - dcol[colindex - kcolstride] * acol) * divided;
                }
            }

            int m_iblocksize, m_jblocksize;
        };

    } // knl

} // namespace platform
